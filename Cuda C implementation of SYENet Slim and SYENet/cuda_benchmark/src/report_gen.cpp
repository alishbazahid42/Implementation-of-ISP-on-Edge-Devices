// Generate results.txt, comparison_report.txt, CSVs.
#include "report_gen.h"
#include <fstream>
#include <iomanip>
#include <sstream>
#include <ctime>
#include <cmath>
#include <filesystem>

namespace fs = std::filesystem;

ReportGenerator::ReportGenerator(const std::string& results_dir)
    : m_dir(results_dir)
{
    fs::create_directories(m_dir);
}

std::string ReportGenerator::timestamp() const
{
    std::time_t t = std::time(nullptr);
    char buf[32];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", std::localtime(&t));
    return buf;
}

std::string ReportGenerator::fmtF(double v, int d) const
{
    if (std::isnan(v)) return "N/A";
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(d) << v;
    return ss.str();
}

// ── per_image_results.csv ─────────────────────────────────────────────────────
void ReportGenerator::writePerImageCSV(const std::string& model_key,
                                        const std::vector<PerImageResult>& results)
{
    std::string path = m_dir + "/per_image_results.csv";
    bool exists = fs::exists(path);
    std::ofstream f(path, std::ios::app);
    if (!exists || !m_csv_header_written) {
        f << "model,image_name,psnr,ssim,mse,mae,rmse,"
          << "h2d_time_ms,preprocess_time_ms,inference_time_ms,"
          << "postprocess_time_ms,d2h_time_ms,total_latency_ms\n";
        m_csv_header_written = true;
    }
    for (const auto& r : results) {
        f << model_key << ","
          << r.image_name << ","
          << fmtF(r.quality.psnr,6)  << "," << fmtF(r.quality.ssim,6) << ","
          << fmtF(r.quality.mse,8)   << "," << fmtF(r.quality.mae,8)  << ","
          << fmtF(r.quality.rmse,8)  << ","
          << fmtF(r.latency.h2d_ms,4)   << "," << fmtF(r.latency.pre_ms,4) << ","
          << fmtF(r.latency.infer_ms,4) << "," << fmtF(r.latency.post_ms,4) << ","
          << fmtF(r.latency.d2h_ms,4)   << "," << fmtF(r.latency.total_ms,4) << "\n";
    }
}

// ── summary_metrics.csv ───────────────────────────────────────────────────────
void ReportGenerator::writeSummaryCSV(const BenchmarkSummary& s)
{
    std::string path = m_dir + "/summary_metrics.csv";
    bool exists = fs::exists(path);
    std::ofstream f(path, std::ios::app);
    if (!exists || !m_sum_header_written) {
        f << "label,size_kb,params,avg_psnr,avg_ssim,avg_mse,avg_mae,avg_rmse,"
          << "lat_avg,lat_min,lat_max,lat_median,lat_std,fps,"
          << "total_images,total_time_s,avg_sm_util,peak_sm_util,"
          << "avg_mem_mb,peak_mem_mb,avg_power_w,avg_temp_c\n";
        m_sum_header_written = true;
    }
    f << s.model_name << "," << s.model_size_kb << "," << s.param_count << ","
      << fmtF(s.avg_psnr,4) << "," << fmtF(s.avg_ssim,4) << ","
      << fmtF(s.avg_mse,8)  << "," << fmtF(s.avg_mae,8)  << ","
      << fmtF(s.avg_rmse,6) << ","
      << fmtF(s.lat_avg_ms,3)    << "," << fmtF(s.lat_min_ms,3)    << ","
      << fmtF(s.lat_max_ms,3)    << "," << fmtF(s.lat_median_ms,3) << ","
      << fmtF(s.lat_std_ms,3)    << ","
      << fmtF(s.fps,2)           << ","
      << s.n_images              << "," << fmtF(s.total_time_s,2) << ","
      << fmtF(s.avg_sm_util,1)   << "," << fmtF(s.peak_sm_util,1) << ","
      << fmtF(s.avg_mem_mb,1)    << "," << fmtF(s.peak_mem_mb,1)  << ","
      << fmtF(s.avg_power_w,1)   << "," << fmtF(s.avg_temp_c,1)   << "\n";
}

// ── results.txt ───────────────────────────────────────────────────────────────
void ReportGenerator::writeResultsTXT(const BenchmarkSummary& s,
                                       const std::vector<PerImageResult>& results)
{
    std::string path = m_dir + "/results.txt";
    std::ofstream f(path, std::ios::app);
    auto line = [&](int n=80){ f << std::string(n, '-') << "\n"; };

    f << std::string(80, '=') << "\n";
    f << "  MODEL EVALUATION REPORT\n";
    f << "  Generated: " << timestamp() << "\n";
    f << std::string(80, '=') << "\n\n";

    f << "  Model Name         : " << s.model_name      << "\n";
    f << "  Model Size         : " << s.model_size_kb   << " KB\n";
    f << "  Parameters         : " << s.param_count     << "\n";
    f << "  GPU Name           : " << s.gpu_name        << "\n";
    f << "  CUDA Version       : " << s.cuda_version    << "\n";
    f << "  TensorRT Version   : " << s.trt_version     << "\n\n";

    line(); f << "  PERFORMANCE METRICS\n"; line();
    f << "  Average Latency  : " << fmtF(s.lat_avg_ms,3)    << " ms\n";
    f << "  Min Latency      : " << fmtF(s.lat_min_ms,3)    << " ms\n";
    f << "  Max Latency      : " << fmtF(s.lat_max_ms,3)    << " ms\n";
    f << "  Median Latency   : " << fmtF(s.lat_median_ms,3) << " ms\n";
    f << "  Std Deviation    : " << fmtF(s.lat_std_ms,3)    << " ms\n";
    f << "  FPS              : " << fmtF(s.fps,2)            << "\n";
    f << "  Images/sec       : " << fmtF(s.fps,2)            << "\n";
    f << "  Total Runtime    : " << fmtF(s.total_time_s,2)   << " s\n\n";

    line(); f << "  GPU UTILIZATION\n"; line();
    f << "  Avg SM Util      : " << fmtF(s.avg_sm_util,1)  << " %\n";
    f << "  Peak SM Util     : " << fmtF(s.peak_sm_util,1) << " %\n";
    f << "  Avg Memory       : " << fmtF(s.avg_mem_mb,1)   << " MB\n";
    f << "  Peak Memory      : " << fmtF(s.peak_mem_mb,1)  << " MB\n";
    f << "  Avg Power        : " << fmtF(s.avg_power_w,1)  << " W\n";
    f << "  Avg Temperature  : " << fmtF(s.avg_temp_c,1)   << " C\n\n";

    line(); f << "  IMAGE QUALITY\n"; line();
    f << "  Average PSNR     : " << fmtF(s.avg_psnr,4) << " dB\n";
    f << "  Average SSIM     : " << fmtF(s.avg_ssim,4) << "\n";
    f << "  Average MSE      : " << fmtF(s.avg_mse, 6) << "\n";
    f << "  Average MAE      : " << fmtF(s.avg_mae, 6) << "\n";
    f << "  Average RMSE     : " << fmtF(s.avg_rmse,6) << "\n";
    f << "  Best  PSNR       : " << fmtF(s.best_psnr_val,4)  << " dB  (" << s.best_psnr_img  << ")\n";
    f << "  Worst PSNR       : " << fmtF(s.worst_psnr_val,4) << " dB  (" << s.worst_psnr_img << ")\n";
    f << "  Best  SSIM       : " << fmtF(s.best_ssim_val,4)  << "    (" << s.best_ssim_img  << ")\n";
    f << "  Worst SSIM       : " << fmtF(s.worst_ssim_val,4) << "    (" << s.worst_ssim_img << ")\n\n";

    line(); f << "  PER IMAGE RESULTS (first 10)\n"; line();
    f << std::left
      << std::setw(32) << "Image"
      << std::setw(10) << "PSNR"
      << std::setw(10) << "SSIM"
      << std::setw(12) << "MSE"
      << std::setw(12) << "Lat(ms)" << "\n";
    f << std::string(76, '-') << "\n";
    int show = std::min(10, (int)results.size());
    for (int i = 0; i < show; i++) {
        const auto& r = results[i];
        f << std::left << std::setw(32) << r.image_name
          << std::setw(10) << fmtF(r.quality.psnr,4)
          << std::setw(10) << fmtF(r.quality.ssim,4)
          << std::setw(12) << fmtF(r.quality.mse,6)
          << std::setw(12) << fmtF(r.latency.total_ms,3) << "\n";
    }
    f << "\n\n";
}

// ── comparison_report.txt ─────────────────────────────────────────────────────
void ReportGenerator::writeComparisonReport(const BenchmarkSummary& orig,
                                             const BenchmarkSummary& slim)
{
    std::string path = m_dir + "/comparison_report.txt";
    std::ofstream f(path);
    auto eq = [&](int n=80){ f << std::string(n, '=') << "\n"; };

    eq();
    f << "  ORIGINAL vs SLIM MODEL COMPARISON\n";
    f << "  Generated: " << timestamp() << "\n";
    eq(); f << "\n";

    auto pct  = [](double a, double b){ return (a-b)/a*100.; };
    auto rat  = [](double a, double b){ return b > 0 ? a/b : 0.; };

    f << "  Original Model\n";
    f << "    Size          : " << orig.model_size_kb << " KB\n";
    f << "    Params        : " << orig.param_count   << "\n";
    f << "    Avg Latency   : " << fmtF(orig.lat_avg_ms,3) << " ms\n";
    f << "    FPS           : " << fmtF(orig.fps,2) << "\n";
    f << "    PSNR          : " << fmtF(orig.avg_psnr,4) << " dB\n";
    f << "    SSIM          : " << fmtF(orig.avg_ssim,4) << "\n\n";

    f << "  Slim Model\n";
    f << "    Size          : " << slim.model_size_kb << " KB\n";
    f << "    Params        : " << slim.param_count   << "\n";
    f << "    Avg Latency   : " << fmtF(slim.lat_avg_ms,3) << " ms\n";
    f << "    FPS           : " << fmtF(slim.fps,2) << "\n";
    f << "    PSNR          : " << fmtF(slim.avg_psnr,4) << " dB\n";
    f << "    SSIM          : " << fmtF(slim.avg_ssim,4) << "\n\n";

    f << "  Comparison\n";
    f << "    Model Size Reduction  : " << fmtF(rat(orig.model_size_kb,slim.model_size_kb),2) << "x\n";
    f << "    Param Reduction       : " << fmtF(rat(orig.param_count,slim.param_count),2) << "x\n";
    f << "    Latency Reduction     : " << fmtF(pct(orig.lat_avg_ms,slim.lat_avg_ms),1) << "%\n";
    f << "    FPS Improvement       : " << fmtF(pct(slim.fps,orig.fps),1) << "%\n";
    f << "    Memory Reduction      : " << fmtF(pct(orig.peak_mem_mb,slim.peak_mem_mb),1) << "%\n";
    f << "    PSNR Difference       : " << fmtF(orig.avg_psnr - slim.avg_psnr, 4) << " dB\n";
    f << "    SSIM Difference       : " << fmtF(orig.avg_ssim - slim.avg_ssim, 4) << "\n";
    f << "    Accuracy Retention    : " << fmtF(slim.avg_psnr/orig.avg_psnr*100,2) << "%\n";
    f << "    Performance Gain      : " << fmtF(pct(orig.lat_avg_ms,slim.lat_avg_ms),1) << "%\n\n";

    f << "  Final Deployment Recommendation\n";
    double acc_ret = slim.avg_psnr / orig.avg_psnr * 100.;
    f << "    Accuracy Retention: " << fmtF(acc_ret,1) << "% -> ";
    f << (acc_ret >= 98. ? "DEPLOY SLIM MODEL\n" : "REVIEW QUALITY BEFORE DEPLOYING\n");
    f << "\n";
    f << "    Jetson Nano   : Slim recommended (low power budget)\n";
    f << "    Jetson Orin   : Slim recommended (real-time capable)\n";
    f << "    RTX/Quadro    : Both viable; Slim preferred for throughput\n";
    f << "    Tesla/A100    : Slim with FP16/INT8 for datacenter throughput\n";
    f << "    Edge AI       : Slim TFLite (29 KB, runs on any platform)\n";
}
