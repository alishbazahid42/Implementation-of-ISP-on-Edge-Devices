// Entry point: validation (CPU vs GPU) + benchmarking (CUDA events).
//
// Usage:
//   sr_infer [H] [W] [iters] [--input input.bin] [--output output.bin]
//
// Defaults: H=360 W=640 iters=200 (640x360 4-ch input -> 1280x720 RGB).
#include "model.h"
#include "utils.h"
#include <cstring>
#include <cstdio>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    int H = 360, W = 640, iters = 200;
    const char *in_path = nullptr, *out_path = nullptr;
    int pos = 0;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--input"))  in_path  = argv[++i];
        else if (!strcmp(argv[i], "--output")) out_path = argv[++i];
        else if (pos == 0) H = atoi(argv[i]), ++pos;
        else if (pos == 1) W = atoi(argv[i]), ++pos;
        else if (pos == 2) iters = atoi(argv[i]), ++pos;
    }
    const size_t n_in = (size_t)4 * H * W, n_out = (size_t)3 * 4 * H * W;

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("Device: %s (SM %d.%d, %d SMs)\n", prop.name, prop.major, prop.minor,
           prop.multiProcessorCount);
    printf("Input: 1x4x%dx%d -> Output: 1x3x%dx%d\n\n", H, W, 2 * H, 2 * W);

    // ---- input -----------------------------------------------------------
    std::vector<float> h_in(n_in), h_out(n_out), h_ref(n_out);
    if (in_path) {
        if (!read_bin(in_path, h_in, n_in)) { fprintf(stderr, "bad input file\n"); return 1; }
    } else {
        fill_random(h_in, 42);
    }

    // ---- model + device buffers ------------------------------------------
    SRModel model(H, W);
    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in,  n_in  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, n_out * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in.data(), n_in * sizeof(float), cudaMemcpyHostToDevice));

    // ---- validation: CPU reference vs GPU ---------------------------------
    printf("Running CPU reference (float64 accumulation)...\n");
    cpu_forward(h_in.data(), h_ref.data(), H, W);
    model.forward(d_in, d_out);
    CUDA_CHECK(cudaMemcpy(h_out.data(), d_out, n_out * sizeof(float), cudaMemcpyDeviceToHost));
    Metrics m = compare(h_ref.data(), h_out.data(), n_out);
    printf("Validation (CPU fp64 ref vs CUDA fp32):\n");
    printf("  MSE          = %.3e\n  MAE          = %.3e\n", m.mse, m.mae);
    printf("  PSNR         = %.2f dB\n  Max abs err  = %.3e\n\n", m.psnr, m.max_abs_err);

    if (out_path) write_bin(out_path, h_out.data(), n_out);

    // ---- per-layer latency -------------------------------------------------
    LayerTimes t{};
    for (int i = 0; i < 10; ++i) model.forward(d_in, d_out, &t);  // warm
    printf("Per-layer latency (timed run; streams serialized by event waits):\n");
    printf("  head.block1 (5x5+PReLU+3x3) : %8.3f ms\n", t.head_block1);
    printf("  head.block2 (5x5)           : %8.3f ms\n", t.head_block2);
    printf("  head merge (+bias)          : %8.3f ms\n", t.head_merge);
    printf("  body conv3x3+ReLU           : %8.3f ms\n", t.body_conv3);
    printf("  body conv1x1+residual       : %8.3f ms\n", t.body_conv1_res);
    printf("  attention (GAP+SE MLP)      : %8.3f ms\n", t.att);
    printf("  channel scale               : %8.3f ms\n", t.channel_scale);
    printf("  pixel shuffle x2            : %8.3f ms\n", t.pixel_shuffle);
    printf("  tail conv3x3                : %8.3f ms\n", t.tail_conv);
    printf("  TOTAL (timed)               : %8.3f ms\n\n", t.total);

    // ---- end-to-end throughput (no per-layer events) -----------------------
    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0)); CUDA_CHECK(cudaEventCreate(&e1));
    for (int i = 0; i < 20; ++i) model.forward(d_in, d_out);      // warm
    CUDA_CHECK(cudaEventRecord(e0));
    for (int i = 0; i < iters; ++i) model.forward(d_in, d_out);
    CUDA_CHECK(cudaEventRecord(e1));
    CUDA_CHECK(cudaEventSynchronize(e1));
    float ms; CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
    ms /= iters;

    size_t free_b, total_b;
    CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
    double flops = ((1200.0 + 1296 + 1200 + 1296 + 144) * H * W +
                    81.0 * 4 * H * W) * 2.0;   // MAC*2; attention negligible
    printf("End-to-end: %.3f ms/frame  =  %.1f FPS\n", ms, 1000.0 / ms);
    printf("Effective compute: %.2f GFLOP/s (model = %.0f MFLOPs/frame)\n",
           flops / ms / 1e6, flops / 1e6);
    printf("Memory: weights %.1f KB (constant), activations %.2f MB, "
           "GPU used %.0f MB / %.0f MB\n",
           SRModel::weight_bytes() / 1024.0, model.activation_bytes() / 1048576.0,
           (total_b - free_b) / 1048576.0, total_b / 1048576.0);

    cudaFree(d_in); cudaFree(d_out);
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    return 0;
}
