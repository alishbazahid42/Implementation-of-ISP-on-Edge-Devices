#pragma once
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t _e = (call);                                               \
        if (_e != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__,   \
                    __LINE__, cudaGetErrorString(_e));                         \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

// ---- CPU reference implementation (float64 accumulation for a tight bound)
// Full network forward on host; layout NCHW, batch 1.
void cpu_forward(const float* x, float* y, int H, int W);

// ---- Error metrics between two buffers of length n
struct Metrics { double mse, mae, psnr, max_abs_err; };
Metrics compare(const float* ref, const float* test, size_t n, float peak = 1.0f);

// ---- Misc
void fill_random(std::vector<float>& v, unsigned seed);
bool read_bin(const char* path, std::vector<float>& v, size_t expect);
bool write_bin(const char* path, const float* p, size_t n);
