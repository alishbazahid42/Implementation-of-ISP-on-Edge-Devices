#pragma once
#include <cuda_runtime.h>
#include <string>

struct ImageMetrics {
    double psnr  = 0.;
    double ssim  = 0.;
    double mse   = 0.;
    double mae   = 0.;
    double rmse  = 0.;
    std::string image_name;
};

struct LatencyMetrics {
    float h2d_ms    = 0.f;
    float pre_ms    = 0.f;
    float infer_ms  = 0.f;
    float post_ms   = 0.f;
    float d2h_ms    = 0.f;
    float total_ms  = 0.f;
};

// GPU parallel PSNR / SSIM kernel wrappers
// input arrays are device pointers, float32, HWC layout
void compute_mse_gpu(const float* d_pred, const float* d_gt,
                     int H, int W, int C,
                     double* h_mse_out,
                     cudaStream_t stream = 0);

void compute_mae_gpu(const float* d_pred, const float* d_gt,
                     int H, int W, int C,
                     double* h_mae_out,
                     cudaStream_t stream = 0);

// CPU-side PSNR / SSIM computed from host arrays (fallback)
double psnr_cpu(const float* pred, const float* gt, int n_pixels, int C);
double ssim_cpu(const float* pred, const float* gt, int H, int W, int C);
