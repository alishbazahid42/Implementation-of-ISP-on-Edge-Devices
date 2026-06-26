#pragma once
#include <string>
#include <cmath>

struct GPUSnapshot {
    unsigned int sm_util    = 0;   // %
    unsigned int mem_util   = 0;   // %
    size_t mem_used_mb      = 0;
    size_t mem_total_mb     = 0;
    unsigned int temp_c     = 0;
    float power_w           = NAN;
    bool valid              = false;
};

class NVMLMonitor {
public:
    NVMLMonitor(int device_idx = 0);
    ~NVMLMonitor();

    bool      available() const { return m_available; }
    GPUSnapshot snapshot();
    std::string gpuName() const { return m_gpu_name; }
    std::string cudaVersion() const { return m_cuda_ver; }
    size_t      totalMemMB() const { return m_total_mem_mb; }

private:
    bool        m_available = false;
    std::string m_gpu_name;
    std::string m_cuda_ver;
    size_t      m_total_mem_mb = 0;
    void*       m_handle = nullptr;   // nvmlDevice_t stored as void*
};
