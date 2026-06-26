// CUDA kernels for per-image quality metrics (MSE, MAE, PSNR, SSIM).
#include "metrics.h"
#include <cuda_runtime.h>
#include <cmath>
#include <cassert>
#include <cstring>

// ── Reduction kernel: squared diff ───────────────────────────────────────────
__global__ void k_mse(const float* __restrict__ pred,
                      const float* __restrict__ gt,
                      double* partial, int n)
{
    extern __shared__ double sdata[];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    double acc = 0.0;
    while (gid < n) {
        double d = (double)pred[gid] - (double)gt[gid];
        acc += d * d;
        gid += gridDim.x * blockDim.x;
    }
    sdata[tid] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) partial[blockIdx.x] = sdata[0];
}

// ── Reduction kernel: absolute diff ──────────────────────────────────────────
__global__ void k_mae(const float* __restrict__ pred,
                      const float* __restrict__ gt,
                      double* partial, int n)
{
    extern __shared__ double sdata[];
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + tid;
    double acc = 0.0;
    while (gid < n) {
        acc += fabs((double)pred[gid] - (double)gt[gid]);
        gid += gridDim.x * blockDim.x;
    }
    sdata[tid] = acc;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    if (tid == 0) partial[blockIdx.x] = sdata[0];
}

// ── Helper: launch reduction and copy result to host ─────────────────────────
static double reduce_gpu(const float* d_pred, const float* d_gt, int n,
                         bool is_mae, cudaStream_t stream)
{
    const int threads = 256;
    const int blocks  = min((n + threads - 1) / threads, 1024);

    double* d_partial;
    cudaMalloc(&d_partial, blocks * sizeof(double));

    size_t shared = threads * sizeof(double);
    if (is_mae)
        k_mae<<<blocks, threads, shared, stream>>>(d_pred, d_gt, d_partial, n);
    else
        k_mse<<<blocks, threads, shared, stream>>>(d_pred, d_gt, d_partial, n);

    std::vector<double> h_partial(blocks);
    cudaMemcpyAsync(h_partial.data(), d_partial, blocks * sizeof(double),
                    cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    cudaFree(d_partial);

    double total = 0.;
    for (double v : h_partial) total += v;
    return total / n;
}

void compute_mse_gpu(const float* d_pred, const float* d_gt,
                     int H, int W, int C, double* h_mse_out, cudaStream_t stream)
{
    *h_mse_out = reduce_gpu(d_pred, d_gt, H * W * C, false, stream);
}

void compute_mae_gpu(const float* d_pred, const float* d_gt,
                     int H, int W, int C, double* h_mae_out, cudaStream_t stream)
{
    *h_mae_out = reduce_gpu(d_pred, d_gt, H * W * C, true, stream);
}

// ── CPU PSNR ─────────────────────────────────────────────────────────────────
double psnr_cpu(const float* pred, const float* gt, int n_pixels, int C)
{
    double mse = 0.;
    int n = n_pixels * C;
    for (int i = 0; i < n; i++) {
        double d = (double)pred[i] - (double)gt[i];
        mse += d * d;
    }
    mse /= n;
    if (mse < 1e-10) return 100.0;
    return 10. * log10(1.0 / mse);
}

// ── CPU SSIM (11×11 Gaussian, per-channel) ───────────────────────────────────
static void gaussian_blur(const double* src, double* dst, int H, int W,
                          double sigma = 1.5, int ks = 11)
{
    // Build 1-D kernel
    int half = ks / 2;
    std::vector<double> k(ks);
    double sum = 0.;
    for (int i = 0; i < ks; i++) {
        double x = i - half;
        k[i] = exp(-x*x / (2.*sigma*sigma));
        sum += k[i];
    }
    for (auto& v : k) v /= sum;

    // Row pass
    std::vector<double> tmp(H * W, 0.);
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            double acc = 0.;
            for (int kx = 0; kx < ks; kx++) {
                int sx = x + kx - half;
                if (sx < 0) sx = 0;
                if (sx >= W) sx = W - 1;
                acc += src[y * W + sx] * k[kx];
            }
            tmp[y * W + x] = acc;
        }
    }
    // Col pass
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            double acc = 0.;
            for (int ky = 0; ky < ks; ky++) {
                int sy = y + ky - half;
                if (sy < 0) sy = 0;
                if (sy >= H) sy = H - 1;
                acc += tmp[sy * W + x] * k[ky];
            }
            dst[y * W + x] = acc;
        }
    }
}

double ssim_cpu(const float* pred, const float* gt, int H, int W, int C)
{
    const double C1 = 0.01 * 0.01;
    const double C2 = 0.03 * 0.03;
    int n = H * W;

    double ssim_sum = 0.;
    std::vector<double> i1(n), i2(n);
    std::vector<double> mu1(n), mu2(n);
    std::vector<double> i1sq(n), i2sq(n), i12(n);
    std::vector<double> s1(n), s2(n), s12(n);

    for (int c = 0; c < C; c++) {
        // Extract channel (HWC → planar)
        for (int p = 0; p < n; p++) {
            i1[p] = pred[p * C + c];
            i2[p] =   gt[p * C + c];
        }
        gaussian_blur(i1.data(), mu1.data(), H, W);
        gaussian_blur(i2.data(), mu2.data(), H, W);

        for (int p = 0; p < n; p++) {
            i1sq[p] = i1[p] * i1[p];
            i2sq[p] = i2[p] * i2[p];
            i12[p]  = i1[p] * i2[p];
        }
        gaussian_blur(i1sq.data(), s1.data(), H, W);
        gaussian_blur(i2sq.data(), s2.data(), H, W);
        gaussian_blur(i12.data(),  s12.data(), H, W);

        double channel_ssim = 0.;
        for (int p = 0; p < n; p++) {
            double m1 = mu1[p], m2 = mu2[p];
            double v1 = s1[p]  - m1*m1;
            double v2 = s2[p]  - m2*m2;
            double v12= s12[p] - m1*m2;
            double num = (2.*m1*m2 + C1) * (2.*v12 + C2);
            double den = (m1*m1 + m2*m2 + C1) * (v1 + v2 + C2);
            channel_ssim += num / den;
        }
        ssim_sum += channel_ssim / n;
    }
    return ssim_sum / C;
}
