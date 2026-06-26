// NVML GPU monitor: SM utilization, memory, power, temperature.
#include "nvml_monitor.h"
#include <nvml.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>

NVMLMonitor::NVMLMonitor(int device_idx)
{
    // CUDA version
    int cudaVer = 0;
    cudaDriverGetVersion(&cudaVer);
    char buf[32];
    snprintf(buf, sizeof(buf), "%d.%d", cudaVer / 1000, (cudaVer % 1000) / 10);
    m_cuda_ver = buf;

    nvmlReturn_t ret = nvmlInit();
    if (ret != NVML_SUCCESS) {
        fprintf(stderr, "[NVML] Init failed: %s\n", nvmlErrorString(ret));
        return;
    }

    nvmlDevice_t dev;
    ret = nvmlDeviceGetHandleByIndex(device_idx, &dev);
    if (ret != NVML_SUCCESS) {
        fprintf(stderr, "[NVML] GetHandle failed: %s\n", nvmlErrorString(ret));
        return;
    }
    m_handle = new nvmlDevice_t(dev);

    char name[96] = {};
    nvmlDeviceGetName(dev, name, sizeof(name));
    m_gpu_name = name;

    nvmlMemory_t mem;
    nvmlDeviceGetMemoryInfo(dev, &mem);
    m_total_mem_mb = mem.total / (1024 * 1024);

    m_available = true;
}

NVMLMonitor::~NVMLMonitor()
{
    if (m_available) nvmlShutdown();
    delete static_cast<nvmlDevice_t*>(m_handle);
}

GPUSnapshot NVMLMonitor::snapshot()
{
    GPUSnapshot s;
    if (!m_available) return s;

    nvmlDevice_t dev = *static_cast<nvmlDevice_t*>(m_handle);

    nvmlUtilization_t util;
    if (nvmlDeviceGetUtilizationRates(dev, &util) == NVML_SUCCESS) {
        s.sm_util  = util.gpu;
        s.mem_util = util.memory;
    }

    nvmlMemory_t mem;
    if (nvmlDeviceGetMemoryInfo(dev, &mem) == NVML_SUCCESS) {
        s.mem_used_mb  = mem.used  / (1024 * 1024);
        s.mem_total_mb = mem.total / (1024 * 1024);
    }

    unsigned int temp = 0;
    if (nvmlDeviceGetTemperature(dev, NVML_TEMPERATURE_GPU, &temp) == NVML_SUCCESS)
        s.temp_c = temp;

    unsigned int power_mw = 0;
    if (nvmlDeviceGetPowerUsage(dev, &power_mw) == NVML_SUCCESS)
        s.power_w = static_cast<float>(power_mw) / 1000.f;

    s.valid = true;
    return s;
}
