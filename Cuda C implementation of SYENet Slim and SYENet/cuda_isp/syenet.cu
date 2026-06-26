/*
 * syenet.cu  —  Standalone CUDA C inference for SYEISPNetS (slim model)
 *
 * Architecture (post-reparameterization):
 *   input (4,128,128) RGGB float32
 *   head:  QCU  4→12  (Conv5x5→PReLU→Conv3x3) * (Conv5x5) + bias
 *   body:  QCU 12→12  (Conv3x3) * (Conv1x1) + bias
 *   att:   SE  GAP → FC → PReLU → FC → Sigmoid → scale
 *   tail:  PixelShuffle(×2) → Conv3x3  → (3,256,256) RGB
 *   clamp [0,1]
 *
 * Weights: 5640 floats (22.1 KB) loaded into __constant__ memory.
 * Input/Output: raw float32 binary files (CHW layout).
 *
 * Compile:  nvcc -O3 -arch=sm_75 syenet.cu -o syenet_isp.exe
 * Run:      syenet_isp.exe weights.bin input.bin output.bin
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

/* ── Fixed network dimensions ─────────────────────────────────────────────── */
#define H_IN   128
#define W_IN   128
#define C_IN   4
#define C_MID  12
#define C_OUT  3
#define H_OUT  256   /* H_IN * 2 */
#define W_OUT  256   /* W_IN * 2 */

#define TOTAL_WEIGHTS 5640

/* ── Weight layout in c_w[] (must match extract_weights.py order) ──────────
   Offset  Key                          Shape          Count  Cumulative
   ------  ---------------------------  -------------  -----  ---------- */
#define OFF_H_B10_W    0      /* head.block1.0.weight   (12,4,5,5)   1200  1200 */
#define OFF_H_B10_B 1200      /* head.block1.0.bias     (12,)          12  1212 */
#define OFF_H_B11_A 1212      /* head.block1.1.weight   (12,) PReLU    12  1224 */
#define OFF_H_B12_W 1224      /* head.block1.2.weight   (12,12,3,3) 1296  2520 */
#define OFF_H_B12_B 2520      /* head.block1.2.bias     (12,)          12  2532 */
#define OFF_H_B2_W  2532      /* head.block2.weight     (12,4,5,5)  1200  3732 */
#define OFF_H_B2_B  3732      /* head.block2.bias       (12,)          12  3744 */
#define OFF_H_BIAS  3744      /* head.bias              (12,)          12  3756 */
#define OFF_BO1_W   3756      /* body.block1.weight     (12,12,3,3) 1296  5052 */
#define OFF_BO1_B   5052      /* body.block1.bias       (12,)          12  5064 */
#define OFF_BO2_W   5064      /* body.block2.weight     (12,12,1,1)  144  5208 */
#define OFF_BO2_B   5208      /* body.block2.bias       (12,)          12  5220 */
#define OFF_BO_BIAS 5220      /* body.bias              (12,)          12  5232 */
#define OFF_AT1_W   5232      /* att.1.weight           (12,12,1,1)  144  5376 */
#define OFF_AT1_B   5376      /* att.1.bias             (12,)          12  5388 */
#define OFF_AT2_A   5388      /* att.2.weight           (12,) PReLU    12  5400 */
#define OFF_AT3_W   5400      /* att.3.weight           (12,12,1,1)  144  5544 */
#define OFF_AT3_B   5544      /* att.3.bias             (12,)          12  5556 */
#define OFF_TL1_W   5556      /* tail.1.weight          (3,3,3,3)     81  5637 */
#define OFF_TL1_B   5637      /* tail.1.bias            (3,)           3  5640 */

__constant__ float c_w[TOTAL_WEIGHTS];

/* ── CUDA error check ─────────────────────────────────────────────────────── */
#define CUDA_CHECK(x) do {                                              \
    cudaError_t _e = (x);                                              \
    if (_e != cudaSuccess) {                                           \
        fprintf(stderr, "CUDA error: %s  at %s:%d\n",                 \
                cudaGetErrorString(_e), __FILE__, __LINE__);           \
        exit(1);                                                       \
    }                                                                  \
} while(0)

/* ══════════════════════════════════════════════════════════════════════════ *
 *  KERNELS
 * ══════════════════════════════════════════════════════════════════════════ */

/*
 * conv2d  —  generic 2-D convolution (NCHW, batch=1, stride=1)
 *
 *   in  shape: (C_in,  H, W)
 *   out shape: (C_out, H, W)
 *   weight: c_w[w_off], shape (C_out, C_in, KH, KW)
 *   bias:   c_w[b_off], shape (C_out,)
 *
 *   Grid:  ( ceil(W/16), ceil(H/16), C_out )
 *   Block: ( 16, 16, 1 )
 */
__global__ void conv2d(
    const float* __restrict__ in,
    float*       __restrict__ out,
    int w_off, int b_off,
    int C_in, int C_out,
    int H, int W,
    int KH, int KW,
    int pad_h, int pad_w)
{
    int ow = blockIdx.x * 16 + threadIdx.x;
    int oh = blockIdx.y * 16 + threadIdx.y;
    int co = blockIdx.z;
    if (ow >= W || oh >= H || co >= C_out) return;

    float acc = c_w[b_off + co];
    int stride_co = C_in * KH * KW;

    for (int ci = 0; ci < C_in; ++ci) {
        int in_base = ci * H * W;
        int w_base  = w_off + co * stride_co + ci * KH * KW;
        for (int kh = 0; kh < KH; ++kh) {
            int ih = oh + kh - pad_h;
            if (ih < 0 || ih >= H) continue;
            for (int kw = 0; kw < KW; ++kw) {
                int iw = ow + kw - pad_w;
                if (iw < 0 || iw >= W) continue;
                acc += in[in_base + ih * W + iw]
                     * c_w[w_base + kh * KW + kw];
            }
        }
    }
    out[co * H * W + oh * W + ow] = acc;
}

/*
 * prelu  —  per-channel parametric ReLU on (C, HW) tensor
 *
 *   Grid:  ceil(C*HW / 256)   Block: 256
 */
__global__ void prelu(float* data, int alpha_off, int C, int HW)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= C * HW) return;
    int c = idx / HW;
    float v = data[idx];
    data[idx] = (v >= 0.f) ? v : c_w[alpha_off + c] * v;
}

/*
 * mul_bias  —  out[i] = a[i] * b[i] + bias[channel]
 *   Tensor layout (C, HW) → channel = idx / HW
 *
 *   Grid:  ceil(C*HW / 256)   Block: 256
 */
__global__ void mul_bias(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float*       __restrict__ out,
    int bias_off, int C, int HW)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= C * HW) return;
    int c = idx / HW;
    out[idx] = a[idx] * b[idx] + c_w[bias_off + c];
}

/*
 * gap  —  global average pool: (C, H, W) → (C,)
 *   One block per channel, one thread, sequential loop.
 *
 *   Grid: (C,)   Block: (1,)
 */
__global__ void gap(const float* __restrict__ in, float* __restrict__ out,
                    int C, int H, int W)
{
    int c = blockIdx.x;
    if (c >= C) return;
    float s = 0.f;
    int HW = H * W;
    const float* row = in + c * HW;
    for (int i = 0; i < HW; ++i) s += row[i];
    out[c] = s / (float)HW;
}

/*
 * fc1x1  —  1×1 conv on a (C_in,) vector → (C_out,) vector
 *   Equivalent to a matrix-vector multiply for SE attention.
 *   weight: c_w[w_off], shape (C_out, C_in)  (from (C_out,C_in,1,1))
 *   bias:   c_w[b_off], shape (C_out,)
 *
 *   Grid: (1,)  Block: (C_out,)  [called with C_out=12]
 */
__global__ void fc1x1(
    const float* __restrict__ in,
    float*       __restrict__ out,
    int w_off, int b_off,
    int C_in, int C_out)
{
    int co = threadIdx.x;
    if (co >= C_out) return;
    float acc = c_w[b_off + co];
    for (int ci = 0; ci < C_in; ++ci)
        acc += in[ci] * c_w[w_off + co * C_in + ci];
    out[co] = acc;
}

/*
 * sigmoid_ip  —  in-place sigmoid
 *   Grid: (1,)  Block: (N,)   [called with N=C_MID=12]
 */
__global__ void sigmoid_ip(float* data, int N)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < N) data[i] = 1.f / (1.f + expf(-data[i]));
}

/*
 * channel_scale  —  x[c, h, w] *= scale[c]   (in-place)
 *   Grid: ceil(C*HW / 256)  Block: 256
 */
__global__ void channel_scale(float* data, const float* __restrict__ scale,
                              int C, int HW)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= C * HW) return;
    data[idx] *= scale[idx / HW];
}

/*
 * pixel_shuffle  —  (C_in=C*4, H, W) → (C, 2H, 2W)
 *
 *   PyTorch formula (scale r=2):
 *     out[c, oh, ow] = in[ c*4 + (oh%2)*2 + (ow%2),  oh/2,  ow/2 ]
 *
 *   Grid:  ( ceil(W_out/16), ceil(H_out/16), C_out )
 *   Block: ( 16, 16, 1 )
 */
__global__ void pixel_shuffle(
    const float* __restrict__ in,
    float*       __restrict__ out,
    int C_out, int H_out, int W_out)
{
    int ow = blockIdx.x * 16 + threadIdx.x;
    int oh = blockIdx.y * 16 + threadIdx.y;
    int co = blockIdx.z;
    if (ow >= W_out || oh >= H_out || co >= C_out) return;

    int H_in = H_out >> 1, W_in = W_out >> 1;
    int c_in = co * 4 + (oh & 1) * 2 + (ow & 1);
    int h_in = oh >> 1, w_in = ow >> 1;

    out[co * H_out * W_out + oh * W_out + ow] =
        in[c_in * H_in * W_in + h_in * W_in + w_in];
}

/*
 * clamp01  —  in-place clamp to [0, 1]
 */
__global__ void clamp01(float* data, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float v = data[i];
        data[i] = (v < 0.f) ? 0.f : ((v > 1.f) ? 1.f : v);
    }
}

/* ══════════════════════════════════════════════════════════════════════════ *
 *  FORWARD PASS
 * ══════════════════════════════════════════════════════════════════════════ */
static void forward(
    float* d_x0,   /* input:  (C_IN,  H_IN,  W_IN)  */
    float* d_b1,   /* work:   (C_MID, H_IN,  W_IN)  */
    float* d_b2,   /* work:   (C_MID, H_IN,  W_IN)  */
    float* d_x1,   /* work:   (C_MID, H_IN,  W_IN)  */
    float* d_gap,  /* work:   (C_MID,)               */
    float* d_ps,   /* work:   (C_OUT, H_OUT, W_OUT)  */
    float* d_out)  /* output: (C_OUT, H_OUT, W_OUT)  */
{
    const int H = H_IN, W = W_IN, C = C_MID, HW = H * W;

    dim3 g_mid( (W+15)/16, (H+15)/16, C );
    dim3 g_out( (W_OUT+15)/16, (H_OUT+15)/16, C_OUT );
    dim3 b16x16(16, 16, 1);
    dim3 b256(256);

    /* ── HEAD ──────────────────────────────────────────────────────────── */
    /* block1: Conv5x5(4→12) + bias → PReLU → Conv3x3(12→12) + bias */
    conv2d<<<g_mid, b16x16>>>(d_x0, d_b1,
        OFF_H_B10_W, OFF_H_B10_B, C_IN, C, H, W, 5, 5, 2, 2);
    prelu<<<((C*HW)+255)/256, b256>>>(d_b1, OFF_H_B11_A, C, HW);
    conv2d<<<g_mid, b16x16>>>(d_b1, d_x1,
        OFF_H_B12_W, OFF_H_B12_B, C, C, H, W, 3, 3, 1, 1);

    /* block2: Conv5x5(4→12) + bias */
    conv2d<<<g_mid, b16x16>>>(d_x0, d_b2,
        OFF_H_B2_W, OFF_H_B2_B, C_IN, C, H, W, 5, 5, 2, 2);

    /* out = block1(x) * block2(x) + head.bias */
    mul_bias<<<((C*HW)+255)/256, b256>>>(d_x1, d_b2, d_b1, OFF_H_BIAS, C, HW);
    /* d_b1 = head output (C_MID, H_IN, W_IN) */

    /* ── BODY ──────────────────────────────────────────────────────────── */
    /* block1: Conv3x3(12→12)  block2: Conv1x1(12→12) */
    conv2d<<<g_mid, b16x16>>>(d_b1, d_x1,
        OFF_BO1_W, OFF_BO1_B, C, C, H, W, 3, 3, 1, 1);
    conv2d<<<g_mid, b16x16>>>(d_b1, d_b2,
        OFF_BO2_W, OFF_BO2_B, C, C, H, W, 1, 1, 0, 0);
    mul_bias<<<((C*HW)+255)/256, b256>>>(d_x1, d_b2, d_b1, OFF_BO_BIAS, C, HW);
    /* d_b1 = body output (C_MID, H_IN, W_IN) */

    /* ── ATT (SE attention) ─────────────────────────────────────────────── */
    /* GAP: (C, H, W) → (C,) */
    gap<<<C, 1>>>(d_b1, d_gap, C, H, W);
    /* FC1: gap → d_b2 (first C elements), weight (12,12,1,1) → linear (12,12) */
    fc1x1<<<1, C>>>(d_gap, d_b2, OFF_AT1_W, OFF_AT1_B, C, C);
    /* PReLU on C-element vector: treat as (C, HW=1) */
    prelu<<<1, C>>>(d_b2, OFF_AT2_A, C, 1);
    /* FC2: d_b2 → d_gap */
    fc1x1<<<1, C>>>(d_b2, d_gap, OFF_AT3_W, OFF_AT3_B, C, C);
    /* Sigmoid */
    sigmoid_ip<<<1, C>>>(d_gap, C);
    /* Scale: x[c,h,w] *= att[c] */
    channel_scale<<<((C*HW)+255)/256, b256>>>(d_b1, d_gap, C, HW);
    /* d_b1 = SE-attended output */

    /* ── TAIL ──────────────────────────────────────────────────────────── */
    /* PixelShuffle×2: (12, 128, 128) → (3, 256, 256) */
    pixel_shuffle<<<g_out, b16x16>>>(d_b1, d_ps, C_OUT, H_OUT, W_OUT);
    /* Conv3x3(3→3) + bias */
    conv2d<<<g_out, b16x16>>>(d_ps, d_out,
        OFF_TL1_W, OFF_TL1_B, C_OUT, C_OUT, H_OUT, W_OUT, 3, 3, 1, 1);
    /* Clamp [0, 1] */
    clamp01<<<(C_OUT*H_OUT*W_OUT+255)/256, b256>>>(d_out, C_OUT*H_OUT*W_OUT);
}

/* ══════════════════════════════════════════════════════════════════════════ *
 *  MAIN
 * ══════════════════════════════════════════════════════════════════════════ */
int main(int argc, char** argv)
{
    const char* weights_path = (argc > 1) ? argv[1] : "weights.bin";
    const char* input_path   = (argc > 2) ? argv[2] : "input.bin";
    const char* output_path  = (argc > 3) ? argv[3] : "output.bin";
    const int   N_ITER_ARG   = (argc > 4) ? atoi(argv[4]) : 100;

    /* Print GPU info */
    int dev = 0;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    printf("GPU: %s  (SM %d.%d, %.0f MB)\n",
           prop.name, prop.major, prop.minor,
           prop.totalGlobalMem / 1048576.0);

    /* ── Load weights into constant memory ─────────────────────────────── */
    printf("Loading weights: %s\n", weights_path);
    FILE* f = fopen(weights_path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", weights_path); return 1; }
    float h_w[TOTAL_WEIGHTS];
    size_t nr = fread(h_w, sizeof(float), TOTAL_WEIGHTS, f);
    fclose(f);
    if ((int)nr != TOTAL_WEIGHTS) {
        fprintf(stderr, "Expected %d floats, got %zu\n", TOTAL_WEIGHTS, nr);
        return 1;
    }
    CUDA_CHECK(cudaMemcpyToSymbol(c_w, h_w, TOTAL_WEIGHTS * sizeof(float)));
    printf("  %d floats (%.1f KB) copied to __constant__ memory\n",
           TOTAL_WEIGHTS, TOTAL_WEIGHTS * 4.f / 1024.f);

    /* ── Load input ─────────────────────────────────────────────────────── */
    printf("Loading input:   %s  (expected %d floats = %d×%d×%d CHW)\n",
           input_path, C_IN * H_IN * W_IN, C_IN, H_IN, W_IN);
    int in_size = C_IN * H_IN * W_IN;
    float* h_in = (float*)malloc(in_size * sizeof(float));
    f = fopen(input_path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", input_path); return 1; }
    fread(h_in, sizeof(float), in_size, f);
    fclose(f);

    /* ── Allocate device buffers ─────────────────────────────────────────── */
    float *d_x0, *d_b1, *d_b2, *d_x1, *d_gap, *d_ps, *d_out_d;
    CUDA_CHECK(cudaMalloc(&d_x0,   in_size         * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b1,   C_MID*H_IN*W_IN * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b2,   C_MID*H_IN*W_IN * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_x1,   C_MID*H_IN*W_IN * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gap,  C_MID           * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ps,   C_OUT*H_OUT*W_OUT * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out_d,C_OUT*H_OUT*W_OUT * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_x0, h_in, in_size * sizeof(float),
                          cudaMemcpyHostToDevice));

    /* ── Warmup (skip if single-pass mode) ────────────────────────────── */
    int n_warmup = (N_ITER_ARG >= 10) ? 5 : 0;
    if (n_warmup > 0) {
        printf("Warmup (%d runs)...\n", n_warmup);
        for (int i = 0; i < n_warmup; i++)
            forward(d_x0, d_b1, d_b2, d_x1, d_gap, d_ps, d_out_d);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    /* ── Timed benchmark ─────────────────────────────────────────────────── */
    cudaEvent_t ev0, ev1;
    CUDA_CHECK(cudaEventCreate(&ev0));
    CUDA_CHECK(cudaEventCreate(&ev1));

    const int N_ITER = N_ITER_ARG;
    CUDA_CHECK(cudaEventRecord(ev0));
    for (int i = 0; i < N_ITER; i++)
        forward(d_x0, d_b1, d_b2, d_x1, d_gap, d_ps, d_out_d);
    CUDA_CHECK(cudaEventRecord(ev1));
    CUDA_CHECK(cudaEventSynchronize(ev1));

    float ms_total = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms_total, ev0, ev1));
    float ms_avg = ms_total / N_ITER;

    if (N_ITER >= 10) {
        printf("\n");
        printf("╔══════════════════════════════════════════╗\n");
        printf("║  SYEISPNetS  CUDA inference results      ║\n");
        printf("╠══════════════════════════════════════════╣\n");
        printf("║  Iterations:       %4d                  ║\n", N_ITER);
        printf("║  Avg latency:  %7.3f ms                ║\n", ms_avg);
        printf("║  Throughput:   %7.1f FPS               ║\n", 1000.f / ms_avg);
        printf("╚══════════════════════════════════════════╝\n");
    } else {
        printf("LATENCY_MS=%.4f\n", ms_avg);
    }

    /* ── Copy output & save ──────────────────────────────────────────────── */
    int out_size = C_OUT * H_OUT * W_OUT;
    float* h_out = (float*)malloc(out_size * sizeof(float));
    CUDA_CHECK(cudaMemcpy(h_out, d_out_d, out_size * sizeof(float),
                          cudaMemcpyDeviceToHost));

    f = fopen(output_path, "wb");
    if (!f) { fprintf(stderr, "Cannot open %s for write\n", output_path); return 1; }
    fwrite(h_out, sizeof(float), out_size, f);
    fclose(f);
    printf("Output saved: %s  (CHW: %d×%d×%d = %d floats)\n",
           output_path, C_OUT, H_OUT, W_OUT, out_size);

    /* ── Cleanup ────────────────────────────────────────────────────────── */
    free(h_in); free(h_out);
    cudaFree(d_x0); cudaFree(d_b1); cudaFree(d_b2); cudaFree(d_x1);
    cudaFree(d_gap); cudaFree(d_ps); cudaFree(d_out_d);
    cudaEventDestroy(ev0); cudaEventDestroy(ev1);

    printf("Run verify.py to compute PSNR vs PyTorch reference.\n");
    return 0;
}
