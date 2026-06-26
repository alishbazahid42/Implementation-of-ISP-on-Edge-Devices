// SYENet ISP CUDA Benchmarking Framework — main entry point.
// Builds TRT engines from ONNX (if not already built), then benchmarks
// original and slim models, and generates all reports.
//
// Usage:
//   ./syenet_bench [--engines_dir <path>] [--inp_dir <path>] [--gt_dir <path>]
//                 [--results_dir <path>] [--warmup <N>] [--fp16] [--int8]
//
// Prerequisites:
//   1. Run tools/01_export_onnx.py to generate engines/original_model.onnx
//      and engines/slim_model.onnx
//   2. CMake build:  cmake -S . -B build && cmake --build build --config Release

#include <cuda_runtime.h>
#include <iostream>
#include <string>
#include <filesystem>
#include "benchmark.h"
#include "report_gen.h"
#include "nvml_monitor.h"
#include "logger.h"

namespace fs = std::filesystem;

static std::string g_engines_dir = R"(C:\Users\aujla\Desktop\archive\cuda_benchmark\engines)";
static std::string g_inp_dir     = R"(C:\Users\aujla\Desktop\archive\dataset\test\mediatek_raw)";
static std::string g_gt_dir      = R"(C:\Users\aujla\Desktop\archive\dataset\test\fujifilm)";
static std::string g_results_dir = R"(C:\Users\aujla\Desktop\archive\cuda_benchmark\results)";
static int         g_warmup      = 20;
static Precision   g_prec        = Precision::FP32;

static void parse_args(int argc, char** argv)
{
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--engines_dir" && i+1 < argc) g_engines_dir = argv[++i];
        else if (a == "--inp_dir"     && i+1 < argc) g_inp_dir     = argv[++i];
        else if (a == "--gt_dir"      && i+1 < argc) g_gt_dir      = argv[++i];
        else if (a == "--results_dir" && i+1 < argc) g_results_dir = argv[++i];
        else if (a == "--warmup"      && i+1 < argc) g_warmup      = std::stoi(argv[++i]);
        else if (a == "--fp16")  g_prec = Precision::FP16;
        else if (a == "--int8")  g_prec = Precision::INT8;
    }
}

static bool ensure_engine(const std::string& onnx, const std::string& eng, Precision prec)
{
    if (fs::exists(eng)) {
        LOG("Engine found: " << eng);
        return true;
    }
    if (!fs::exists(onnx)) {
        LOG_ERR("ONNX not found: " << onnx);
        LOG_ERR("Run tools/01_export_onnx.py first.");
        return false;
    }
    TRTEngine builder;
    return builder.buildFromONNX(onnx, eng, prec);
}

int main(int argc, char** argv)
{
    parse_args(argc, argv);

    // ── GPU info ──────────────────────────────────────────────────────────────
    int dev_count = 0;
    cudaGetDeviceCount(&dev_count);
    if (dev_count == 0) { LOG_ERR("No CUDA device found."); return 1; }
    cudaSetDevice(0);

    NVMLMonitor monitor(0);
    LOG("GPU     : " << monitor.gpuName());
    LOG("CUDA    : " << monitor.cudaVersion());
    LOG("Mem     : " << monitor.totalMemMB() << " MB");
    LOG("Prec    : " << (g_prec == Precision::FP16 ? "FP16" :
                         g_prec == Precision::INT8  ? "INT8" : "FP32"));

    std::string prec_suffix = (g_prec == Precision::FP16 ? "_fp16" :
                               g_prec == Precision::INT8  ? "_int8" : "_fp32");

    // ── Engine paths ──────────────────────────────────────────────────────────
    std::string orig_onnx = g_engines_dir + "/original_model.onnx";
    std::string slim_onnx = g_engines_dir + "/slim_model.onnx";
    std::string orig_eng  = g_engines_dir + "/original" + prec_suffix + ".engine";
    std::string slim_eng  = g_engines_dir + "/slim"     + prec_suffix + ".engine";

    if (!ensure_engine(orig_onnx, orig_eng, g_prec)) return 1;
    if (!ensure_engine(slim_onnx, slim_eng, g_prec)) return 1;

    ReportGenerator reporter(g_results_dir);

    // ── Benchmark original ────────────────────────────────────────────────────
    LOG("\n==== BENCHMARKING: Original Model ====");
    BenchmarkConfig cfg_orig;
    cfg_orig.model_name    = "Original SYEISPNet (" + std::string(g_prec == Precision::FP16 ? "FP16" : "FP32") + ")";
    cfg_orig.engine_path   = orig_eng;
    cfg_orig.inp_dir       = g_inp_dir;
    cfg_orig.gt_dir        = g_gt_dir;
    cfg_orig.results_dir   = g_results_dir;
    cfg_orig.warmup_images = g_warmup;
    cfg_orig.precision     = g_prec;
    cfg_orig.model_size_kb = fs::exists(R"(C:\Users\aujla\Desktop\archive\model_best.pkl)")
                             ? fs::file_size(R"(C:\Users\aujla\Desktop\archive\model_best.pkl)") / 1024 : 0;

    std::vector<PerImageResult> results_orig;
    BenchmarkSummary summary_orig;
    Benchmark bench_orig(cfg_orig, monitor);
    if (!bench_orig.run(results_orig, summary_orig)) {
        LOG_ERR("Original model benchmark failed"); return 1;
    }
    summary_orig.model_size_kb = cfg_orig.model_size_kb;
    summary_orig.param_count   = 110984;
    summary_orig.trt_version   = NV_TENSORRT_MAJOR "." NV_TENSORRT_MINOR "." NV_TENSORRT_PATCH;

    reporter.writePerImageCSV("original", results_orig);
    reporter.writeSummaryCSV(summary_orig);
    reporter.writeResultsTXT(summary_orig, results_orig);

    // ── Benchmark slim ────────────────────────────────────────────────────────
    LOG("\n==== BENCHMARKING: Slim Model ====");
    BenchmarkConfig cfg_slim = cfg_orig;
    cfg_slim.model_name    = "Slim SYEISPNetS (" + std::string(g_prec == Precision::FP16 ? "FP16" : "FP32") + ")";
    cfg_slim.engine_path   = slim_eng;
    cfg_slim.model_size_kb = fs::exists(R"(C:\Users\aujla\Desktop\archive\slim_model.pt)")
                             ? fs::file_size(R"(C:\Users\aujla\Desktop\archive\slim_model.pt)") / 1024 : 0;

    std::vector<PerImageResult> results_slim;
    BenchmarkSummary summary_slim;
    Benchmark bench_slim(cfg_slim, monitor);
    if (!bench_slim.run(results_slim, summary_slim)) {
        LOG_ERR("Slim model benchmark failed"); return 1;
    }
    summary_slim.model_size_kb = cfg_slim.model_size_kb;
    summary_slim.param_count   = 5640;
    summary_slim.trt_version   = NV_TENSORRT_MAJOR "." NV_TENSORRT_MINOR "." NV_TENSORRT_PATCH;

    reporter.writePerImageCSV("slim", results_slim);
    reporter.writeSummaryCSV(summary_slim);
    reporter.writeResultsTXT(summary_slim, results_slim);
    reporter.writeComparisonReport(summary_orig, summary_slim);

    // ── Final console summary ─────────────────────────────────────────────────
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "FINAL BENCHMARK SUMMARY\n";
    std::cout << std::string(60, '=') << "\n\n";

    for (const auto* s : {&summary_orig, &summary_slim}) {
        std::cout << "  " << s->model_name << "\n";
        std::cout << "    PSNR    : " << s->avg_psnr   << " dB\n";
        std::cout << "    SSIM    : " << s->avg_ssim   << "\n";
        std::cout << "    Latency : " << s->lat_avg_ms << " ms\n";
        std::cout << "    FPS     : " << s->fps        << "\n\n";
    }

    double speedup = summary_orig.lat_avg_ms / summary_slim.lat_avg_ms;
    double acc_ret = summary_slim.avg_psnr   / summary_orig.avg_psnr * 100.;
    std::cout << "  Speedup           : " << speedup << "x\n";
    std::cout << "  Accuracy Retention: " << acc_ret << "%\n";
    std::cout << "\nResults in: " << g_results_dir << "\n";
    return 0;
}
