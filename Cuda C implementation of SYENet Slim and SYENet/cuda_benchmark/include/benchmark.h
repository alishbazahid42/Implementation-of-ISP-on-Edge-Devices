#pragma once
#include <cuda_runtime.h>
#include <string>
#include <vector>
#include "trt_engine.h"
#include "metrics.h"
#include "nvml_monitor.h"

struct BenchmarkConfig {
    std::string model_name;
    std::string engine_path;
    std::string inp_dir;
    std::string gt_dir;
    std::string results_dir;
    int  warmup_images  = 20;
    int  batch_size     = 1;
    bool save_outputs   = false;
    Precision precision = Precision::FP32;
};

struct PerImageResult {
    std::string  image_name;
    ImageMetrics quality;
    LatencyMetrics latency;
    GPUSnapshot  gpu_snap;
};

struct BenchmarkSummary {
    std::string  model_name;
    size_t       model_size_kb = 0;
    int64_t      param_count   = 0;
    int          n_images      = 0;

    // Latency stats
    double lat_avg_ms = 0., lat_min_ms = 0., lat_max_ms = 0.;
    double lat_median_ms = 0., lat_std_ms = 0.;
    double inf_avg_ms = 0.;
    double fps = 0., total_time_s = 0.;

    // Quality stats
    double avg_psnr = 0., avg_ssim = 0.;
    double avg_mse  = 0., avg_mae  = 0., avg_rmse = 0.;
    std::string best_psnr_img,  worst_psnr_img;
    double      best_psnr_val = 0., worst_psnr_val = 1e9;
    std::string best_ssim_img,  worst_ssim_img;
    double      best_ssim_val = 0., worst_ssim_val = 1e9;

    // GPU stats
    double avg_sm_util = 0., peak_sm_util = 0.;
    double avg_mem_mb  = 0., peak_mem_mb  = 0.;
    double avg_power_w = 0., avg_temp_c   = 0.;
    std::string gpu_name, cuda_version, trt_version;
};

class Benchmark {
public:
    Benchmark(const BenchmarkConfig& cfg, NVMLMonitor& monitor);
    ~Benchmark();

    bool run(std::vector<PerImageResult>& out_results,
             BenchmarkSummary&           out_summary);

private:
    BenchmarkConfig  m_cfg;
    NVMLMonitor&     m_monitor;
    TRTEngine        m_engine;

    // Pinned host buffers + device buffers
    float*   h_input  = nullptr;
    float*   h_output = nullptr;
    float*   d_input  = nullptr;
    float*   d_output = nullptr;
    float*   d_gt     = nullptr;
    size_t   m_inp_bytes = 0;
    size_t   m_out_bytes = 0;

    cudaStream_t m_stream = nullptr;

    bool allocateBuffers();
    void freeBuffers();

    // Bayer → RGGB unpack + normalise (CPU)
    bool loadRaw(const std::string& path,
                 float* h_inp,        // output: (4, H/2, W/2) float32
                 int* out_H, int* out_W);

    // Load ground-truth RGB PNG into float32 HWC [0,1]
    bool loadGT(const std::string& path,
                float* h_gt,          // output: (H, W, 3)
                int* out_H, int* out_W);
};
