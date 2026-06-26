"""
Step 2: Build TensorRT engines from ONNX models.
Generates FP32, FP16 engines for both original and slim models.
INT8 is skipped (requires calibration dataset setup).

Requirements:
  pip install tensorrt  (or install TensorRT from NVIDIA developer zone)
  TensorRT 8.x / 10.x with CUDA 12
"""
import os, sys, time
import numpy as np

ENGINES_DIR = r'C:\Users\aujla\Desktop\archive\cuda_benchmark\engines'
DATASET_DIR = r'C:\Users\aujla\Desktop\archive\dataset\test\mediatek_raw'

def build_engine(onnx_path, engine_path, fp16=False, int8=False, calib_data=None):
    try:
        import tensorrt as trt
    except ImportError:
        print("TensorRT not installed. Skipping engine build.")
        print("Install from: https://developer.nvidia.com/tensorrt")
        return False

    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder    = trt.Builder(TRT_LOGGER)
    network    = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser     = trt.OnnxParser(network, TRT_LOGGER)
    config     = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1 GB

    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print(f"  FP16 enabled")
    if int8 and builder.platform_has_fast_int8:
        config.set_flag(trt.BuilderFlag.INT8)
        print(f"  INT8 enabled")

    with open(onnx_path, 'rb') as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"  TRT parse error: {parser.get_error(i)}")
            return False

    print(f"  Building engine (this may take 1-3 min) ...")
    t0 = time.time()
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        print("  Engine build FAILED")
        return False

    with open(engine_path, 'wb') as f:
        f.write(serialized)
    print(f"  Done in {time.time()-t0:.1f}s  ->  {os.path.getsize(engine_path)/1024:.1f} KB")
    return True


configs = [
    ('original_model.onnx', 'original_fp32.engine', False, False),
    ('original_model.onnx', 'original_fp16.engine', True,  False),
    ('slim_model.onnx',     'slim_fp32.engine',     False, False),
    ('slim_model.onnx',     'slim_fp16.engine',     True,  False),
]

for onnx_name, engine_name, fp16, int8 in configs:
    onnx_path   = os.path.join(ENGINES_DIR, onnx_name)
    engine_path = os.path.join(ENGINES_DIR, engine_name)
    if not os.path.exists(onnx_path):
        print(f"ONNX not found: {onnx_path}  (run 01_export_onnx.py first)")
        continue
    if os.path.exists(engine_path):
        print(f"Engine exists, skipping: {engine_name}")
        continue
    label = 'FP16' if fp16 else 'FP32'
    print(f"\nBuilding {engine_name} ({label}) ...")
    build_engine(onnx_path, engine_path, fp16=fp16)

print("\nEngine build complete.")
