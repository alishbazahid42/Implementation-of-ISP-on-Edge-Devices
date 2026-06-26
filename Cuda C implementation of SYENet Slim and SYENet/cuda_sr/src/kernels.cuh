// CUDA kernels for the recovered 5,640-parameter ×2 super-resolution network.
// All weights live in __constant__ memory (22.5 KB < 64 KB budget): every
// thread in a warp reads the same weight at the same time, so constant-cache
// broadcast makes weight access effectively free.
#pragma once
#include <cuda_runtime.h>

// Fused activation applied inside the conv epilogue.
enum Activation { ACT_NONE = 0, ACT_RELU = 1, ACT_PRELU = 2 };

// Conv tile: 32 threads along x gives fully coalesced 128-byte rows.
#define TILE_W 32
#define TILE_H 8

// Copy all weights into constant memory. Call once at startup.
void upload_weights_to_constant(const float* host_weights, int n_floats);

// Direct conv2d, NCHW, stride 1, pad K/2 ("same"), weights/bias read from
// __constant__ memory at the given float offsets. Shared-memory input tiling.
// prelu_off: offset of per-channel PReLU slopes (only used when act==ACT_PRELU).
template <int K>
void launch_conv2d(const float* in, float* out,
                   int Cin, int Cout, int H, int W,
                   int w_off, int b_off, Activation act, int prelu_off,
                   cudaStream_t stream);

// out[c,y,x] = a[c,y,x] + b[c,y,x] + cbias[c]   (head merge: block1+block2+head.bias)
void launch_add3_bias(const float* a, const float* b, int cbias_off,
                      float* out, int C, int H, int W, cudaStream_t stream);

// out[c,y,x] = x[c,y,x] + res[c,y,x] + cbias[c] (body residual + body.bias)
void launch_residual_bias(const float* x, const float* res, int cbias_off,
                          float* out, int C, int H, int W, cudaStream_t stream);

// Per-channel global average pool: in (C,H,W) -> out (C). One block per channel.
void launch_global_avg_pool(const float* in, float* out, int C, int H, int W,
                            cudaStream_t stream);

// Squeeze-excitation tail on the pooled vector (C=12):
// att = sigmoid(W3 * prelu(W1 * g + b1) + b3). Single-block kernel — the
// matrices are 12x12, this is latency-bound, not throughput-bound.
void launch_attention_fc(const float* pooled, float* att, int C,
                         int w1_off, int b1_off, int prelu_off,
                         int w3_off, int b3_off, cudaStream_t stream);

// out[c,y,x] = in[c,y,x] * att[c]
void launch_channel_scale(const float* in, const float* att, float* out,
                          int C, int H, int W, cudaStream_t stream);

// PixelShuffle r=2: (C*r^2, H, W) -> (C, H*r, W*r). Pure index permutation.
void launch_pixel_shuffle2(const float* in, float* out,
                           int Cout, int H, int W, cudaStream_t stream);
