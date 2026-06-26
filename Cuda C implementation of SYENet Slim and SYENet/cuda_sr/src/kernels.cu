#include "kernels.cuh"
#include "utils.h"
#include <cstdio>

// ---------------------------------------------------------------------------
// Constant memory: the whole model (5,640 floats = 22.56 KB) fits with room
// to spare. Warp-uniform reads hit the constant cache broadcast path.
// ---------------------------------------------------------------------------
__constant__ float c_w[5640];

void upload_weights_to_constant(const float* host_weights, int n_floats) {
    CUDA_CHECK(cudaMemcpyToSymbol(c_w, host_weights, n_floats * sizeof(float)));
}

// ---------------------------------------------------------------------------
// Tiled direct convolution.
// Grid:  (ceil(W/TILE_W), ceil(H/TILE_H), Cout)
// Block: (TILE_W, TILE_H)
// For each input channel we stage a (TILE_H+K-1) x (TILE_W+K-1) halo tile in
// shared memory (coalesced global loads), then every thread accumulates its
// KxK window. Weights come from constant memory (warp-uniform -> broadcast).
// ---------------------------------------------------------------------------
template <int K>
__global__ void conv2d_kernel(const float* __restrict__ in, float* __restrict__ out,
                              int Cin, int Cout, int H, int W,
                              int w_off, int b_off, int act, int prelu_off) {
    constexpr int P = K / 2;
    constexpr int SW = TILE_W + K - 1;
    constexpr int SH = TILE_H + K - 1;
    __shared__ float smem[SH][SW];

    const int ox = blockIdx.x * TILE_W + threadIdx.x;
    const int oy = blockIdx.y * TILE_H + threadIdx.y;
    const int oc = blockIdx.z;
    const int tid = threadIdx.y * TILE_W + threadIdx.x;

    float acc = c_w[b_off + oc];

    for (int ic = 0; ic < Cin; ++ic) {
        // Cooperative halo load (zero padding at borders).
        const float* src = in + (size_t)ic * H * W;
        for (int i = tid; i < SH * SW; i += TILE_W * TILE_H) {
            int sy = i / SW, sx = i % SW;
            int gy = blockIdx.y * TILE_H + sy - P;
            int gx = blockIdx.x * TILE_W + sx - P;
            smem[sy][sx] = (gy >= 0 && gy < H && gx >= 0 && gx < W)
                               ? __ldg(&src[gy * W + gx]) : 0.0f;
        }
        __syncthreads();

        if (ox < W && oy < H) {
            const float* wk = &c_w[w_off + ((oc * Cin + ic) * K * K)];
            #pragma unroll
            for (int ky = 0; ky < K; ++ky)
                #pragma unroll
                for (int kx = 0; kx < K; ++kx)
                    acc += smem[threadIdx.y + ky][threadIdx.x + kx] * wk[ky * K + kx];
        }
        __syncthreads();
    }

    if (ox >= W || oy >= H) return;
    if (act == ACT_RELU)       acc = fmaxf(acc, 0.0f);
    else if (act == ACT_PRELU) acc = acc >= 0.0f ? acc : acc * c_w[prelu_off + oc];
    out[((size_t)oc * H + oy) * W + ox] = acc;
}

template <int K>
void launch_conv2d(const float* in, float* out, int Cin, int Cout, int H, int W,
                   int w_off, int b_off, Activation act, int prelu_off,
                   cudaStream_t stream) {
    dim3 block(TILE_W, TILE_H);
    dim3 grid((W + TILE_W - 1) / TILE_W, (H + TILE_H - 1) / TILE_H, Cout);
    conv2d_kernel<K><<<grid, block, 0, stream>>>(in, out, Cin, Cout, H, W,
                                                 w_off, b_off, act, prelu_off);
}
template void launch_conv2d<1>(const float*, float*, int, int, int, int, int, int, Activation, int, cudaStream_t);
template void launch_conv2d<3>(const float*, float*, int, int, int, int, int, int, Activation, int, cudaStream_t);
template void launch_conv2d<5>(const float*, float*, int, int, int, int, int, int, Activation, int, cudaStream_t);

// ---------------------------------------------------------------------------
// Elementwise merges (memory-bound; grid-stride, coalesced).
// ---------------------------------------------------------------------------
__global__ void add3_bias_kernel(const float* a, const float* b, int cbias_off,
                                 float* out, int C, size_t plane, size_t n) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n;
         i += (size_t)gridDim.x * blockDim.x)
        out[i] = a[i] + b[i] + c_w[cbias_off + (int)(i / plane)];
}

void launch_add3_bias(const float* a, const float* b, int cbias_off, float* out,
                      int C, int H, int W, cudaStream_t stream) {
    size_t n = (size_t)C * H * W;
    add3_bias_kernel<<<(int)((n + 255) / 256), 256, 0, stream>>>(a, b, cbias_off, out, C, (size_t)H * W, n);
}

void launch_residual_bias(const float* x, const float* res, int cbias_off,
                          float* out, int C, int H, int W, cudaStream_t stream) {
    launch_add3_bias(x, res, cbias_off, out, C, H, W, stream); // same math
}

// ---------------------------------------------------------------------------
// Global average pool: one block per channel, tree reduction in shared memory.
// ---------------------------------------------------------------------------
__global__ void gap_kernel(const float* in, float* out, int H, int W) {
    __shared__ float red[256];
    const int c = blockIdx.x;
    const size_t plane = (size_t)H * W;
    const float* src = in + c * plane;
    float s = 0.0f;
    for (size_t i = threadIdx.x; i < plane; i += blockDim.x) s += src[i];
    red[threadIdx.x] = s;
    __syncthreads();
    for (int k = blockDim.x / 2; k > 0; k >>= 1) {
        if (threadIdx.x < k) red[threadIdx.x] += red[threadIdx.x + k];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[c] = red[0] / (float)plane;
}

void launch_global_avg_pool(const float* in, float* out, int C, int H, int W,
                            cudaStream_t stream) {
    gap_kernel<<<C, 256, 0, stream>>>(in, out, H, W);
}

// ---------------------------------------------------------------------------
// SE attention MLP on the pooled 12-vector. One block, C threads. The hidden
// vector is staged in shared memory; total work is 2 x (12x12) MACs.
// ---------------------------------------------------------------------------
__global__ void attention_fc_kernel(const float* pooled, float* att, int C,
                                    int w1_off, int b1_off, int prelu_off,
                                    int w3_off, int b3_off) {
    extern __shared__ float hidden[];
    const int c = threadIdx.x;
    if (c < C) {
        float h = c_w[b1_off + c];
        for (int i = 0; i < C; ++i) h += c_w[w1_off + c * C + i] * pooled[i];
        hidden[c] = h >= 0.0f ? h : h * c_w[prelu_off + c];
    }
    __syncthreads();
    if (c < C) {
        float o = c_w[b3_off + c];
        for (int i = 0; i < C; ++i) o += c_w[w3_off + c * C + i] * hidden[i];
        att[c] = 1.0f / (1.0f + __expf(-o));
    }
}

void launch_attention_fc(const float* pooled, float* att, int C,
                         int w1_off, int b1_off, int prelu_off,
                         int w3_off, int b3_off, cudaStream_t stream) {
    attention_fc_kernel<<<1, 32, C * sizeof(float), stream>>>(
        pooled, att, C, w1_off, b1_off, prelu_off, w3_off, b3_off);
}

__global__ void channel_scale_kernel(const float* in, const float* att,
                                     float* out, size_t plane, size_t n) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n;
         i += (size_t)gridDim.x * blockDim.x)
        out[i] = in[i] * __ldg(&att[(int)(i / plane)]);
}

void launch_channel_scale(const float* in, const float* att, float* out,
                          int C, int H, int W, cudaStream_t stream) {
    size_t n = (size_t)C * H * W;
    channel_scale_kernel<<<(int)((n + 255) / 256), 256, 0, stream>>>(in, att, out, (size_t)H * W, n);
}

// ---------------------------------------------------------------------------
// PixelShuffle r=2: out[c, 2y+dy, 2x+dx] = in[c*4 + dy*2 + dx, y, x].
// Threads indexed over the OUTPUT so writes are coalesced (reads are strided,
// but stay in L2 thanks to the small working set).
// ---------------------------------------------------------------------------
__global__ void pixel_shuffle2_kernel(const float* in, float* out,
                                      int Cout, int H, int W) {
    const int Ho = 2 * H, Wo = 2 * W;
    size_t n = (size_t)Cout * Ho * Wo;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n;
         i += (size_t)gridDim.x * blockDim.x) {
        int x = (int)(i % Wo), y = (int)((i / Wo) % Ho), c = (int)(i / ((size_t)Wo * Ho));
        int ci = c * 4 + (y & 1) * 2 + (x & 1);
        out[i] = in[((size_t)ci * H + (y >> 1)) * W + (x >> 1)];
    }
}

void launch_pixel_shuffle2(const float* in, float* out, int Cout, int H, int W,
                           cudaStream_t stream) {
    size_t n = (size_t)Cout * 4 * H * W;
    pixel_shuffle2_kernel<<<(int)((n + 255) / 256), 256, 0, stream>>>(in, out, Cout, H, W);
}
