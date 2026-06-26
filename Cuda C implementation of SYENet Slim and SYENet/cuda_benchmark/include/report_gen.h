#pragma once
#include <string>
#include <vector>
#include "benchmark.h"

class ReportGenerator {
public:
    explicit ReportGenerator(const std::string& results_dir);

    // Write per_image_results.csv
    void writePerImageCSV(const std::string& model_key,
                          const std::vector<PerImageResult>& results);

    // Write summary_metrics.csv (append one row per model)
    void writeSummaryCSV(const BenchmarkSummary& summary);

    // Write human-readable results.txt
    void writeResultsTXT(const BenchmarkSummary& summary,
                         const std::vector<PerImageResult>& results);

    // Write comparison_report.txt (call after both orig + slim are benchmarked)
    void writeComparisonReport(const BenchmarkSummary& orig,
                               const BenchmarkSummary& slim);

private:
    std::string m_dir;
    bool        m_csv_header_written = false;
    bool        m_sum_header_written = false;

    std::string fmtF(double v, int d = 4) const;
    std::string timestamp() const;
};
