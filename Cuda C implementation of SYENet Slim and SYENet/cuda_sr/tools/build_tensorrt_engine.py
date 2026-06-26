#!/usr/bin/env python3
"""
Build a TensorRT engine from model.onnx and run a quick benchmark.

    python build_tensorrt_engine.py model.onnx [--fp16] [--int8]
                                    [--shape 1x4x360x640] [--engine model.plan]

Works with TensorRT 8.x and 10.x (Jetson Nano = TRT 8.2 via JetPack 4.6,
Orin Nano = TRT 8.5+/10.x via JetPack 5/6). INT8 uses a random-data entropy
calibrator as a placeholder — swap `calib_batches()` for real frames before
trusting INT8 accuracy.
"""
import argparse, os
import numpy as np
import tensorrt as trt

TRT10 = int(trt.__version__.split(".")[0]) >= 10
LOGGER = trt.Logger(trt.Logger.INFO)


class RandomCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, shape, n_batches=32, cache="calib.cache"):
        super().__init__()
        import pycuda.autoinit  # noqa: F401
        import pycuda.driver as cuda
        self.cuda = cuda
        self.shape, self.n, self.i, self.cache = shape, n_batches, 0, cache
        self.dev = cuda.mem_alloc(int(np.prod(shape)) * 4)

    def calib_batches(self):
        # TODO: replace with real input frames (packed Bayer / RGBA, [0,1])
        return np.random.rand(*self.shape).astype(np.float32)

    def get_batch_size(self): return self.shape[0]

    def get_batch(self, names):
        if self.i >= self.n: return None
        self.i += 1
        self.cuda.memcpy_htod(self.dev, np.ascontiguousarray(self.calib_batches()))
        return [int(self.dev)]

    def read_calibration_cache(self):
        return open(self.cache, "rb").read() if os.path.exists(self.cache) else None

    def write_calibration_cache(self, cache):
        open(self.cache, "wb").write(cache)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("onnx")
    ap.add_argument("--engine", default="model.plan")
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--int8", action="store_true")
    ap.add_argument("--shape", default="1x4x360x640")
    args = ap.parse_args()
    shape = tuple(int(d) for d in args.shape.split("x"))

    builder = trt.Builder(LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, LOGGER)
    if not parser.parse(open(args.onnx, "rb").read()):
        for i in range(parser.num_errors):
            print(parser.get_error(i))
        raise SystemExit("ONNX parse failed")

    config = builder.create_builder_config()
    if TRT10:
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 28)
    else:
        config.max_workspace_size = 1 << 28
    if args.fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    if args.int8 and builder.platform_has_fast_int8:
        config.set_flag(trt.BuilderFlag.INT8)
        config.int8_calibrator = RandomCalibrator(shape)

    profile = builder.create_optimization_profile()
    profile.set_shape("input", min=(1, 4, 64, 64), opt=shape,
                      max=(1, 4, shape[2] * 2, shape[3] * 2))
    config.add_optimization_profile(profile)

    print("Building engine (may take a few minutes on Jetson)...")
    blob = builder.build_serialized_network(network, config)
    open(args.engine, "wb").write(blob)
    print(f"Saved {args.engine} ({len(blob)/1024:.0f} KB)")

    # ---- quick benchmark ---------------------------------------------------
    import pycuda.autoinit  # noqa: F401
    import pycuda.driver as cuda
    rt = trt.Runtime(LOGGER)
    engine = rt.deserialize_cuda_engine(blob)
    ctx = engine.create_execution_context()
    out_shape = (1, 3, shape[2] * 2, shape[3] * 2)
    d_in = cuda.mem_alloc(int(np.prod(shape)) * 4)
    d_out = cuda.mem_alloc(int(np.prod(out_shape)) * 4)
    stream = cuda.Stream()

    if TRT10:
        ctx.set_input_shape("input", shape)
        ctx.set_tensor_address("input", int(d_in))
        ctx.set_tensor_address("output", int(d_out))
        run = lambda: ctx.execute_async_v3(stream.handle)
    else:
        ctx.set_binding_shape(0, shape)
        run = lambda: ctx.execute_async_v2([int(d_in), int(d_out)], stream.handle)

    cuda.memcpy_htod(d_in, np.random.rand(*shape).astype(np.float32))
    for _ in range(20): run()
    stream.synchronize()
    start, end = cuda.Event(), cuda.Event()
    start.record(stream)
    for _ in range(200): run()
    end.record(stream)
    end.synchronize()
    ms = start.time_till(end) / 200
    print(f"TensorRT: {ms:.3f} ms/frame = {1000/ms:.1f} FPS "
          f"({'INT8' if args.int8 else 'FP16' if args.fp16 else 'FP32'})")


if __name__ == "__main__":
    main()
