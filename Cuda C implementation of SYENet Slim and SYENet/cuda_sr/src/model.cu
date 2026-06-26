#include "model.h"
#include "kernels.cuh"
#include "weights.h"
#include "utils.h"
#include <vector>

static const int C = 12;     // trunk width
static const int CIN = 4, COUT = 3;

size_t SRModel::weight_bytes() { return TOTAL_WEIGHT_FLOATS * sizeof(float); }

SRModel::SRModel(int H, int W) : H_(H), W_(W) {
    upload_weights_to_constant(MODEL_WEIGHTS, TOTAL_WEIGHT_FLOATS);

    size_t plane = (size_t)H * W;
    auto alloc = [&](float** p, size_t floats) {
        CUDA_CHECK(cudaMalloc(p, floats * sizeof(float)));
        act_bytes_ += floats * sizeof(float);
    };
    alloc(&d_b1_, C * plane);  alloc(&d_b2_, C * plane);  alloc(&d_h_, C * plane);
    alloc(&d_t_,  C * plane);  alloc(&d_b_,  C * plane);  alloc(&d_f_, C * plane);
    alloc(&d_up_, COUT * 4 * plane);
    alloc(&d_pool_, C);        alloc(&d_att_, C);

    CUDA_CHECK(cudaStreamCreate(&s1_));
    CUDA_CHECK(cudaStreamCreate(&s2_));
    CUDA_CHECK(cudaEventCreateWithFlags(&ev_b2_done_, cudaEventDisableTiming));
}

SRModel::~SRModel() {
    for (float* p : {d_b1_, d_b2_, d_h_, d_t_, d_b_, d_f_, d_up_, d_pool_, d_att_})
        cudaFree(p);
    cudaStreamDestroy(s1_); cudaStreamDestroy(s2_);
    cudaEventDestroy(ev_b2_done_);
}

// Small helper for optional per-layer event timing.
struct ScopedTimer {
    cudaEvent_t a{}, b{}; float* out; cudaStream_t s;
    ScopedTimer(float* o, cudaStream_t st) : out(o), s(st) {
        if (out) { cudaEventCreate(&a); cudaEventCreate(&b); cudaEventRecord(a, s); }
    }
    ~ScopedTimer() {
        if (out) {
            cudaEventRecord(b, s); cudaEventSynchronize(b);
            cudaEventElapsedTime(out, a, b);
            cudaEventDestroy(a); cudaEventDestroy(b);
        }
    }
};

void SRModel::forward(const float* d_in, float* d_out, LayerTimes* tm) {
    const int H = H_, W = W_;
    float* nul = nullptr;
    auto T = [&](float LayerTimes::*f) { return tm ? &(tm->*f) : nul; };

    cudaEvent_t e0{}, e1{};
    if (tm) { cudaEventCreate(&e0); cudaEventCreate(&e1); cudaEventRecord(e0, s1_); }

    // --- head: two independent branches on two streams -------------------
    {
        ScopedTimer t(T(&LayerTimes::head_block2), s2_);
        launch_conv2d<5>(d_in, d_b2_, CIN, C, H, W,
                         HEAD_BLOCK2_WEIGHT_OFFSET, HEAD_BLOCK2_BIAS_OFFSET,
                         ACT_NONE, 0, s2_);
        CUDA_CHECK(cudaEventRecord(ev_b2_done_, s2_));
    }
    {
        ScopedTimer t(T(&LayerTimes::head_block1), s1_);
        launch_conv2d<5>(d_in, d_t_, CIN, C, H, W,
                         HEAD_BLOCK1_0_WEIGHT_OFFSET, HEAD_BLOCK1_0_BIAS_OFFSET,
                         ACT_PRELU, HEAD_BLOCK1_1_PRELU_OFFSET, s1_);
        launch_conv2d<3>(d_t_, d_b1_, C, C, H, W,
                         HEAD_BLOCK1_2_WEIGHT_OFFSET, HEAD_BLOCK1_2_BIAS_OFFSET,
                         ACT_NONE, 0, s1_);
    }
    CUDA_CHECK(cudaStreamWaitEvent(s1_, ev_b2_done_, 0));   // join branches
    {
        ScopedTimer t(T(&LayerTimes::head_merge), s1_);
        launch_add3_bias(d_b1_, d_b2_, HEAD_BIAS_OFFSET, d_h_, C, H, W, s1_);
    }

    // --- body -------------------------------------------------------------
    {
        ScopedTimer t(T(&LayerTimes::body_conv3), s1_);
        launch_conv2d<3>(d_h_, d_t_, C, C, H, W,
                         BODY_BLOCK1_WEIGHT_OFFSET, BODY_BLOCK1_BIAS_OFFSET,
                         ACT_RELU, 0, s1_);
    }
    {
        ScopedTimer t(T(&LayerTimes::body_conv1_res), s1_);
        launch_conv2d<1>(d_t_, d_b_, C, C, H, W,
                         BODY_BLOCK2_WEIGHT_OFFSET, BODY_BLOCK2_BIAS_OFFSET,
                         ACT_NONE, 0, s1_);
        launch_residual_bias(d_b_, d_h_, BODY_BIAS_OFFSET, d_b_, C, H, W, s1_);
    }

    // --- attention (squeeze-excitation) ------------------------------------
    {
        ScopedTimer t(T(&LayerTimes::att), s1_);
        launch_global_avg_pool(d_b_, d_pool_, C, H, W, s1_);
        launch_attention_fc(d_pool_, d_att_, C,
                            ATT_1_WEIGHT_OFFSET, ATT_1_BIAS_OFFSET, ATT_2_PRELU_OFFSET,
                            ATT_3_WEIGHT_OFFSET, ATT_3_BIAS_OFFSET, s1_);
    }
    {
        ScopedTimer t(T(&LayerTimes::channel_scale), s1_);
        launch_channel_scale(d_b_, d_att_, d_f_, C, H, W, s1_);
    }

    // --- tail: PixelShuffle(2) -> Conv3x3(3->3) -----------------------------
    {
        ScopedTimer t(T(&LayerTimes::pixel_shuffle), s1_);
        launch_pixel_shuffle2(d_f_, d_up_, COUT, H, W, s1_);
    }
    {
        ScopedTimer t(T(&LayerTimes::tail_conv), s1_);
        launch_conv2d<3>(d_up_, d_out, COUT, COUT, 2 * H, 2 * W,
                         TAIL_1_WEIGHT_OFFSET, TAIL_1_BIAS_OFFSET,
                         ACT_NONE, 0, s1_);
    }

    if (tm) {
        cudaEventRecord(e1, s1_);
        CUDA_CHECK(cudaEventSynchronize(e1));
        cudaEventElapsedTime(&tm->total, e0, e1);
        cudaEventDestroy(e0); cudaEventDestroy(e1);
    } else {
        CUDA_CHECK(cudaStreamSynchronize(s1_));
    }
}
