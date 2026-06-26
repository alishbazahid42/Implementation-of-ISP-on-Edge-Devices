# CUDA Inference for the Recovered ×2 Super-Resolution Network

Everything below was extracted directly from the provided checkpoint
(`data.pkl` + raw storages in `data/`), with **no PyTorch required** — a stub
unpickler (`tools/inspect_model.py`) recovered every tensor's name, shape,
dtype, stride, and storage mapping.

---

## 1. Architecture report (extracted, not assumed)

The checkpoint is a bare `OrderedDict` state_dict (no module object), saved
from `cuda:0`, all tensors `float32`. 20 tensors, **5,640 parameters,
22.56 KB** in FP32.

### Tensor inventory (exact, from data.pkl)

| # | Key | Shape | Params | Interpretation |
|---|-----|-------|--------|----------------|
| 0 | `head.bias` | (1,12,1,1) | 12 | learnable per-channel bias |
| 1 | `head.block1.0.weight` | (12,4,5,5) | 1200 | Conv5×5, 4→12 |
| 2 | `head.block1.0.bias` | (12) | 12 | |
| 3 | `head.block1.1.weight` | (12) | 12 | **PReLU** (learnable per-channel slope) |
| 4 | `head.block1.2.weight` | (12,12,3,3) | 1296 | Conv3×3, 12→12 |
| 5 | `head.block1.2.bias` | (12) | 12 | |
| 6 | `head.block2.weight` | (12,4,5,5) | 1200 | Conv5×5, 4→12 (parallel branch) |
| 7 | `head.block2.bias` | (12) | 12 | |
| 8 | `body.bias` | (1,12,1,1) | 12 | learnable per-channel bias |
| 9 | `body.block1.weight` | (12,12,3,3) | 1296 | Conv3×3, 12→12 |
| 10 | `body.block1.bias` | (12) | 12 | |
| 11 | `body.block2.weight` | (12,12,1,1) | 144 | Conv1×1, 12→12 |
| 12 | `body.block2.bias` | (12) | 12 | |
| 13 | `att.1.weight` | (12,12,1,1) | 144 | Conv1×1 (SE squeeze) |
| 14 | `att.1.bias` | (12) | 12 | |
| 15 | `att.2.weight` | (12) | 12 | **PReLU** |
| 16 | `att.3.weight` | (12,12,1,1) | 144 | Conv1×1 (SE excite) |
| 17 | `att.3.bias` | (12) | 12 | |
| 18 | `tail.1.weight` | (3,3,3,3) | 81 | Conv3×3, 3→3 |
| 19 | `tail.1.bias` | (3) | 3 | |

**Total: 5,640 parameters.**

### Recovered dataflow

```
input x: (1, 4, H, W)            # 4-channel input (packed RGGB Bayer or RGBA)
head:
  b1 = Conv5x5(4→12) → PReLU(12) → Conv3x3(12→12)   # block1
  b2 = Conv5x5(4→12)                                 # block2 (parallel)
  h  = b1 + b2 + head.bias
body:
  t  = ReLU(Conv3x3(12→12, h))                       # activation: see assumptions
  b  = Conv1x1(12→12, t) + body.bias + h             # residual: see assumptions
att (squeeze-excitation):
  a  = Sigmoid(Conv1x1(12→12, PReLU(Conv1x1(12→12, GlobalAvgPool(b)))))
  f  = b ⊙ a                                          # per-channel scale
tail:
  y  = Conv3x3(3→3, PixelShuffle×2(f))   →  (1, 3, 2H, 2W)
```

**Hard facts** (forced by tensor shapes / Sequential indices):
- 4-in / 3-out channels, trunk width 12, ×2 upscale: 12 = 3·2², and
  `tail.0` is parameter-less before a 3→3 conv ⇒ `tail.0 = PixelShuffle(2)`.
- `head.block1.1` and `att.2` are PReLU — their weights are learnable
  per-channel scalars, which BatchNorm/ReLU don't have (no BN anywhere:
  no `running_mean`/`running_var` keys exist).
- `att.0` is parameter-less before two 1×1 convs that operate on 12 channels,
  followed by nothing spatial ⇒ global-average-pool squeeze-excitation block.

**Inferred (the .pkl contains weights only, no graph)** — flagged in code:
- Body activation between block1/block2 assumed **ReLU** (no PReLU weight
  exists there); residual `+ h` and the merge topology `b1+b2+bias` assumed
  from the standard pattern these module names follow. If you have the
  original model class, only `Body.forward`/`Head.forward` in
  `tools/export_to_onnx.py` and the corresponding two launches in
  `src/model.cu` would need adjusting.
- 4-channel input at half resolution with ×2 PixelShuffle to 3-channel RGB is
  the classic **joint demosaic/SR (packed RGGB RAW → RGB)** mobile-ISP layout.

### Compute / memory (per frame, input 1×4×360×640)

| Quantity | Value |
|---|---|
| MACs per input pixel | 5,460 (head 3,696 / body 1,440 / tail 324 / att ≈0) |
| FLOPs @ 640×360 → 1280×720 | **2.52 GFLOPs** |
| Weights | 22.56 KB (fits entirely in `__constant__` memory) |
| Activations (FP32, all buffers) | 9·HW·4 B ≈ **15.8 MB** |
| Validation output | 1×3×720×1280 |

---

## 2. Project layout

```
cuda_sr/
├── CMakeLists.txt
├── src/
│   ├── main.cu          # validation + per-layer & end-to-end benchmark
│   ├── model.h/.cu      # buffers, streams, forward orchestration
│   ├── kernels.cuh/.cu  # conv2d (tiled, templated K=1/3/5, fused act),
│   │                    # GAP, SE-MLP, channel-scale, pixel-shuffle, merges
│   ├── utils.h/.cu      # CPU fp64 reference, MSE/MAE/PSNR/max-err, I/O
│   ├── weights.h        # generated: all 5,640 floats + offset manifest
│   └── weights.bin      # generated: same data as raw binary
└── tools/
    ├── inspect_model.py        # torch-free .pkl analyzer (regenerates report)
    ├── export_weights.py       # data/* → weights.h + weights.bin
    ├── sanity_check.py         # pure-numpy forward + weight-range audit
    ├── export_to_onnx.py       # rebuilds nn.Module, strict load, ONNX export
    └── build_tensorrt_engine.py# ONNX → TRT engine (FP32/FP16/INT8) + bench
```

Per-layer kernels requested in Step 4 map to: `conv2d_kernel<5/3/1>` (conv),
`attention_fc_kernel` + `gap_kernel` + `channel_scale_kernel` (attention,
sigmoid fused), PReLU/ReLU fused into the conv epilogue,
`pixel_shuffle2_kernel`, `add3_bias_kernel` (residual/bias merges). There is
no BatchNorm or pooling layer in the model other than the SE global pool.

---

## 3. Build & run

```bash
cd cuda_sr && mkdir build && cd build
cmake -DCMAKE_CUDA_ARCHITECTURES=87 ..   # 87=Orin Nano, 53=Nano, 86=RTX30xx, 89=RTX40xx
cmake --build . --config Release
./sr_infer 360 640 200                   # H W iters
./sr_infer 360 640 200 --input frame.bin --output sr.bin   # real data (CHW f32)
```

Output: validation block (MSE / MAE / PSNR / max-abs-err vs the CPU float64
reference), per-layer CUDA-event latencies, end-to-end ms/FPS, GFLOP/s, and
memory utilization (`cudaMemGetInfo`).

---

## 4. Optimization strategy (and why)

| Technique | Where | Why it wins here |
|---|---|---|
| **Constant memory for all weights** | `c_w[5640]` in kernels.cu | The whole model is 22.5 KB < 64 KB. Every thread in a warp reads the same weight simultaneously → constant-cache broadcast, zero global traffic for weights. This is the single biggest structural win for a sub-6K-param net. |
| **Shared-memory input tiling** | `conv2d_kernel<K>` | Each input pixel is reused K² times per output channel; staging a (TILE+K−1)² halo tile per input channel turns K²-redundant global reads into one coalesced load + smem reads. |
| **Coalesced access** | all kernels | 32-wide thread rows over NCHW innermost dim → 128-byte transactions. PixelShuffle indexes over the *output* so writes coalesce. |
| **Fused epilogues** | conv + bias + PReLU/ReLU | The net is memory-bound (5.5 KMAC/px vs ~40 B/px traffic); each avoided intermediate tensor round-trip ≈ one whole elementwise kernel saved. |
| **CUDA streams + events** | `head.block1` ∥ `head.block2` | The two head branches are independent; on big GPUs (RTX/Quadro/Tesla) the 4→12 convs don't fill the machine alone, so they overlap. `cudaStreamWaitEvent` joins before the merge. |
| **Async copies** | benchmark path keeps data resident; `cudaMemcpyAsync`-ready stream API throughout | H2D/D2H can overlap the previous frame's compute in a pipelined deployment. |
| **`--use_fast_math`** | sigmoid `__expf` etc. | Attention gate is saturated (observed gates ≈ 0/1), so fast-math exp error is irrelevant. |
| **Tensor Cores** | *deliberately not used in custom kernels* | TCs need ≥8–16 channel tiles to pay off; at C=12/4/3 with 3×3/5×5 direct conv, the fragment padding waste exceeds the gain. The TensorRT path (below) will use TCs automatically where its tactic search finds them profitable (mainly the implicit-GEMM 12→12 convs in FP16 on Orin/RTX). |
| **Jetson notes** | CMake arch list | Nano (SM53) has no TCs and 4 GB shared with CPU — the constant-memory + fused-kernel design is exactly right for it. Orin Nano (SM87): use TRT FP16 for best results; lock clocks with `jetson_clocks` before benchmarking. |

---

## 5. Precision modes

| Mode | Weights | Activations @360×640 | Expected speed vs FP32 | Accuracy impact |
|---|---|---|---|---|
| FP32 (native CUDA, default) | 22.6 KB | 15.8 MB | 1× | bit-exact vs fp64 ref to ~1e-4 (fast-math sigmoid) |
| FP16 (TensorRT `--fp16`) | 11.3 KB | 7.9 MB | ~1.5–2× (memory-bound → ~halved traffic; TCs on 12→12 convs) | PSNR drop typically <0.05 dB for SR nets this small; the saturated sigmoid gates are robust |
| INT8 (TensorRT `--int8`) | 5.6 KB | 4 MB | ~2–3× on Orin/RTX, **not supported on Jetson Nano (SM53)** | ⚠ `head.block1.0.weight` contains a −98.5 outlier (abs-mean 1.43, measured by `tools/sanity_check.py`). Per-tensor quantization would destroy that conv — **per-channel weight quantization (TRT default) is mandatory**, and the random calibrator in the script must be replaced with real frames. |

Realistic end-to-end estimates for 640×360→1280×720 (model is tiny; launch
overhead and memory traffic dominate): RTX 3060+ ≥2,000 FPS FP32; Orin Nano
~400–800 FPS FP16-TRT; Jetson Nano ~30–60 FPS FP32 (its ~25 GB/s bandwidth is
the wall). Measure with the built-in benchmark — these are sizing estimates.

---

## 6. ONNX / TensorRT

```bash
pip install torch onnx onnxruntime        # x86; on Jetson use NVIDIA wheels
python tools/export_to_onnx.py --height 360 --width 640 --out model.onnx
python tools/build_tensorrt_engine.py model.onnx --fp16          # engine + bench
python tools/build_tensorrt_engine.py model.onnx --int8          # see caveat above
# or with trtexec:
trtexec --onnx=model.onnx --fp16 --saveEngine=model.plan \
        --shapes=input:1x4x360x640
```

`export_to_onnx.py` loads weights with `strict=True` (any key/shape mismatch
fails loudly) and cross-checks torch vs onnxruntime output.

---

## 7. Jetson deployment

**Jetson Orin Nano (JetPack 6.x — CUDA 12, cuDNN 9, TensorRT 10 preinstalled):**
```bash
sudo apt update && sudo apt install -y nvidia-jetpack cmake build-essential
sudo nvpmodel -m 0 && sudo jetson_clocks          # max power mode, lock clocks
cd cuda_sr && mkdir build && cd build
cmake -DCMAKE_CUDA_ARCHITECTURES=87 .. && make -j$(nproc)
./sr_infer 360 640 500
# TensorRT path (python3-libnvinfer ships with JetPack):
python3 tools/export_to_onnx.py && python3 tools/build_tensorrt_engine.py model.onnx --fp16
```

**Jetson Nano (JetPack 4.6 — CUDA 10.2, cuDNN 8.2, TensorRT 8.2):**
```bash
sudo apt install -y nvidia-jetpack cmake
sudo nvpmodel -m 0 && sudo jetson_clocks
cmake -DCMAKE_CUDA_ARCHITECTURES=53 .. && make -j4
./sr_infer 270 480 200        # smaller frames advised; 4GB unified memory
```
Nano notes: FP16 helps little (Maxwell has no TCs, but half traffic still
helps via `__half2`-capable TRT kernels); INT8 unavailable; export ONNX on a
host PC if torch install on Nano is painful (the export script only needs the
`data/` files, copy them over or run it on the host).

GPU utilization: `tegrastats` (Jetson) or `nvidia-smi dmon` (desktop), or
profile per-kernel with `nsys profile ./sr_infer` / `ncu --set full ./sr_infer`
(the build keeps `-lineinfo` for source-level attribution).
