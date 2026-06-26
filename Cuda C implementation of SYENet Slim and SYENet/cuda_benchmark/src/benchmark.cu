// Core benchmarking loop: async CUDA streams, pinned memory, CUDA event timing.
#include "benchmark.h"
#include <fstream>
#include <filesystem>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <cassert>

// STB image for PNG I/O (header-only; drop stb_image.h in include/)
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

namespace fs = std::filesystem;

// ── Constructor ───────────────────────────────────────────────────────────────
Benchmark::Benchmark(const BenchmarkConfig& cfg, NVMLMonitor& monitor)
    : m_cfg(cfg), m_monitor(monitor)
{
    cudaStreamCreate(&m_stream);
}

Benchmark::~Benchmark()
{
    freeBuffers();
    if (m_stream) cudaStreamDestroy(m_stream);
}

// ── Buffer allocation ─────────────────────────────────────────────────────────
bool Benchmark::allocateBuffers()
{
    // Fixed input: (1, 4, 128, 128) float32
    // Fixed output: (1, 3, 256, 256) float32
    m_inp_bytes = 1 * 4 * 128 * 128 * sizeof(float);
    m_out_bytes = 1 * 3 * 256 * 256 * sizeof(float);
    size_t gt_bytes  = 256 * 256 * 3 * sizeof(float);  // HWC for metrics

    cudaMallocHost(&h_input,  m_inp_bytes);
    cudaMallocHost(&h_output, m_out_bytes);
    cudaMalloc(    &d_input,  m_inp_bytes);
    cudaMalloc(    &d_output, m_out_bytes);
    cudaMalloc(    &d_gt,     gt_bytes);
    return h_input && h_output && d_input && d_output && d_gt;
}

void Benchmark::freeBuffers()
{
    if (h_input)  cudaFreeHost(h_input);
    if (h_output) cudaFreeHost(h_output);
    if (d_input)  cudaFree(d_input);
    if (d_output) cudaFree(d_output);
    if (d_gt)     cudaFree(d_gt);
    h_input = h_output = nullptr;
    d_input = d_output = d_gt = nullptr;
}

// ── RAW PNG loader + Bayer→RGGB unpacking ────────────────────────────────────
bool Benchmark::loadRaw(const std::string& path, float* h_inp, int* oH, int* oW)
{
    // PNG is 16-bit single-channel (Bayer)
    int W, H, ch;
    unsigned short* px = reinterpret_cast<unsigned short*>(
        stbi_load_16(path.c_str(), &W, &H, &ch, 1));
    if (!px) return false;

    *oH = H; *oW = W;
    // Bayer → RGGB: (H,W) → (4, H/2, W/2)
    // R  = [0::2, 0::2], G1 = [0::2, 1::2]
    // G2 = [1::2, 0::2], B  = [1::2, 1::2]
    int hH = H / 2, hW = W / 2;
    float* R  = h_inp + 0 * hH * hW;
    float* G1 = h_inp + 1 * hH * hW;
    float* G2 = h_inp + 2 * hH * hW;
    float* B  = h_inp + 3 * hH * hW;
    for (int y = 0; y < hH; y++) {
        for (int x = 0; x < hW; x++) {
            R [y*hW+x] = px[(2*y  )*W + 2*x  ] / 4095.f;
            G1[y*hW+x] = px[(2*y  )*W + 2*x+1] / 4095.f;
            G2[y*hW+x] = px[(2*y+1)*W + 2*x  ] / 4095.f;
            B [y*hW+x] = px[(2*y+1)*W + 2*x+1] / 4095.f;
        }
    }
    stbi_image_free(px);
    return true;
}

bool Benchmark::loadGT(const std::string& path, float* h_gt, int* oH, int* oW)
{
    int W, H, ch;
    unsigned char* px = stbi_load(path.c_str(), &W, &H, &ch, 3);
    if (!px) return false;
    *oH = H; *oW = W;
    for (int i = 0; i < H * W * 3; i++)
        h_gt[i] = px[i] / 255.f;
    stbi_image_free(px);
    return true;
}

// ── Main benchmark loop ───────────────────────────────────────────────────────
bool Benchmark::run(std::vector<PerImageResult>& out_results,
                    BenchmarkSummary&           out_summary)
{
    if (!m_engine.loadEngine(m_cfg.engine_path)) return false;
    if (!allocateBuffers()) { LOG_ERR("Buffer alloc failed"); return false; }

    // Enumerate image pairs
    std::vector<std::pair<std::string,std::string>> pairs;
    for (auto& e : fs::directory_iterator(m_cfg.inp_dir)) {
        if (e.path().extension() != ".png") continue;
        std::string stem = e.path().stem().string();
        fs::path gt_path = fs::path(m_cfg.gt_dir) / (stem + ".png");
        if (fs::exists(gt_path))
            pairs.push_back({e.path().string(), gt_path.string()});
    }
    std::sort(pairs.begin(), pairs.end(),
              [](auto& a, auto& b){ return a.first < b.first; });

    int N = static_cast<int>(pairs.size());
    LOG("Running " << N << " images  (warmup=" << m_cfg.warmup_images << ")");

    out_results.clear();
    out_results.reserve(N - m_cfg.warmup_images);

    // Temp host buffer for GT
    std::vector<float> h_gt(256 * 256 * 3);
    std::vector<float> h_out(3 * 256 * 256);

    for (int i = 0; i < N; i++) {
        const auto& [inp_path, gt_path] = pairs[i];
        std::string stem = fs::path(inp_path).stem().string();

        int rawH, rawW, gtH, gtW;
        if (!loadRaw(inp_path, h_input, &rawH, &rawW)) {
            LOG_ERR("Failed to load RAW: " << inp_path); continue;
        }
        if (!loadGT(gt_path, h_gt.data(), &gtH, &gtW)) {
            LOG_ERR("Failed to load GT: " << gt_path); continue;
        }

        // CUDA events for each stage
        cudaEvent_t ev[10];
        for (auto& e : ev) cudaEventCreate(&e);

        // ── H2D ──────────────────────────────────────────────────────────────
        cudaEventRecord(ev[0], m_stream);
        cudaMemcpyAsync(d_input, h_input, m_inp_bytes,
                        cudaMemcpyHostToDevice, m_stream);
        cudaEventRecord(ev[1], m_stream);

        // ── Preprocess (no-op: already done on CPU) ───────────────────────────
        cudaEventRecord(ev[2], m_stream);
        cudaEventRecord(ev[3], m_stream);

        // ── Inference ────────────────────────────────────────────────────────
        cudaEventRecord(ev[4], m_stream);
        m_engine.infer(d_input, d_output, m_stream);
        cudaEventRecord(ev[5], m_stream);

        // ── Postprocess: clamp (simple kernel) ───────────────────────────────
        // For simplicity we clamp on CPU after D2H
        cudaEventRecord(ev[6], m_stream);
        cudaEventRecord(ev[7], m_stream);

        // ── D2H ──────────────────────────────────────────────────────────────
        cudaEventRecord(ev[8], m_stream);
        cudaMemcpyAsync(h_output, d_output, m_out_bytes,
                        cudaMemcpyDeviceToHost, m_stream);
        cudaEventRecord(ev[9], m_stream);

        cudaStreamSynchronize(m_stream);

        // Clamp output [0,1]
        int out_n = 3 * 256 * 256;
        for (int j = 0; j < out_n; j++)
            h_out[j] = std::max(0.f, std::min(1.f, h_output[j]));

        // Convert CHW → HWC for metrics
        std::vector<float> h_out_hwc(out_n);
        for (int c = 0; c < 3; c++)
            for (int p = 0; p < 256*256; p++)
                h_out_hwc[p*3+c] = h_out[c*256*256+p];

        auto snap = m_monitor.snapshot();

        if (i >= m_cfg.warmup_images) {
            PerImageResult res;
            res.image_name = stem;
            res.latency.h2d_ms   = 0.f; cudaEventElapsedTime(&res.latency.h2d_ms,   ev[0], ev[1]);
            res.latency.pre_ms   = 0.f; cudaEventElapsedTime(&res.latency.pre_ms,   ev[2], ev[3]);
            res.latency.infer_ms = 0.f; cudaEventElapsedTime(&res.latency.infer_ms, ev[4], ev[5]);
            res.latency.post_ms  = 0.f; cudaEventElapsedTime(&res.latency.post_ms,  ev[6], ev[7]);
            res.latency.d2h_ms   = 0.f; cudaEventElapsedTime(&res.latency.d2h_ms,   ev[8], ev[9]);
            res.latency.total_ms = res.latency.h2d_ms + res.latency.pre_ms +
                                   res.latency.infer_ms + res.latency.post_ms +
                                   res.latency.d2h_ms;

            double mse, mae;
            compute_mse_gpu(d_output, d_gt, 256, 256, 3, &mse, m_stream);
            compute_mae_gpu(d_output, d_gt, 256, 256, 3, &mae, m_stream);
            res.quality.mse   = mse;
            res.quality.mae   = mae;
            res.quality.rmse  = std::sqrt(mse);
            res.quality.psnr  = psnr_cpu(h_out_hwc.data(), h_gt.data(), 256*256, 3);
            res.quality.ssim  = ssim_cpu(h_out_hwc.data(), h_gt.data(), 256, 256, 3);
            res.quality.image_name = stem;
            res.gpu_snap = snap;

            out_results.push_back(res);

            if ((i+1) % 200 == 0)
                LOG("  " << i+1 << "/" << N
                    << "  PSNR=" << res.quality.psnr
                    << "  inf=" << res.latency.infer_ms << " ms");
        }

        for (auto& e : ev) cudaEventDestroy(e);
    }

    // ── Compute summary ───────────────────────────────────────────────────────
    out_summary.model_name = m_cfg.model_name;
    out_summary.n_images   = static_cast<int>(out_results.size());
    out_summary.gpu_name   = m_monitor.gpuName();
    out_summary.cuda_version = m_monitor.cudaVersion();

    std::vector<double> lat_vals;
    for (auto& r : out_results) {
        lat_vals.push_back(r.latency.total_ms);
        out_summary.avg_psnr += r.quality.psnr;
        out_summary.avg_ssim += r.quality.ssim;
        out_summary.avg_mse  += r.quality.mse;
        out_summary.avg_mae  += r.quality.mae;
        out_summary.avg_rmse += r.quality.rmse;
        out_summary.inf_avg_ms += r.latency.infer_ms;

        if (r.quality.psnr > out_summary.best_psnr_val) {
            out_summary.best_psnr_val = r.quality.psnr;
            out_summary.best_psnr_img = r.image_name;
        }
        if (r.quality.psnr < out_summary.worst_psnr_val) {
            out_summary.worst_psnr_val = r.quality.psnr;
            out_summary.worst_psnr_img = r.image_name;
        }
        if (r.quality.ssim > out_summary.best_ssim_val) {
            out_summary.best_ssim_val = r.quality.ssim;
            out_summary.best_ssim_img = r.image_name;
        }
        if (r.quality.ssim < out_summary.worst_ssim_val) {
            out_summary.worst_ssim_val = r.quality.ssim;
            out_summary.worst_ssim_img = r.image_name;
        }
        if (r.gpu_snap.valid) {
            out_summary.avg_sm_util += r.gpu_snap.sm_util;
            out_summary.avg_mem_mb  += r.gpu_snap.mem_used_mb;
            out_summary.avg_power_w += r.gpu_snap.power_w;
            out_summary.avg_temp_c  += r.gpu_snap.temp_c;
            if (r.gpu_snap.sm_util  > out_summary.peak_sm_util)
                out_summary.peak_sm_util = r.gpu_snap.sm_util;
            if (r.gpu_snap.mem_used_mb > out_summary.peak_mem_mb)
                out_summary.peak_mem_mb  = r.gpu_snap.mem_used_mb;
        }
    }
    int M = out_summary.n_images;
    if (M > 0) {
        out_summary.avg_psnr /= M;  out_summary.avg_ssim  /= M;
        out_summary.avg_mse  /= M;  out_summary.avg_mae   /= M;
        out_summary.avg_rmse /= M;  out_summary.inf_avg_ms /= M;
        out_summary.avg_sm_util /= M; out_summary.avg_mem_mb /= M;
        out_summary.avg_power_w /= M; out_summary.avg_temp_c /= M;
    }
    std::sort(lat_vals.begin(), lat_vals.end());
    out_summary.lat_min_ms    = lat_vals.front();
    out_summary.lat_max_ms    = lat_vals.back();
    out_summary.lat_avg_ms    = std::accumulate(lat_vals.begin(), lat_vals.end(), 0.) / M;
    out_summary.lat_median_ms = lat_vals[M / 2];
    double var = 0.;
    for (double v : lat_vals) { double d = v - out_summary.lat_avg_ms; var += d*d; }
    out_summary.lat_std_ms    = std::sqrt(var / M);
    out_summary.fps           = 1000. / out_summary.lat_avg_ms;
    out_summary.total_time_s  = std::accumulate(lat_vals.begin(), lat_vals.end(), 0.) / 1000.;

    freeBuffers();
    return true;
}
