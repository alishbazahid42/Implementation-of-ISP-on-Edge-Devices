// GPU model: buffer management + forward pass for the recovered SR network.
//
// Recovered architecture (from data.pkl — see README.md for the full report):
//   input  x: (1, 4, H, W)            4-ch input (packed RGGB Bayer / RGBA)
//   head:  b1 = Conv5x5(4->12) -> PReLU(12) -> Conv3x3(12->12)
//          b2 = Conv5x5(4->12)                       [parallel branch]
//          h  = b1 + b2 + head.bias                  [per-channel bias]
//   body:  t  = ReLU(Conv3x3(12->12, h))             [activation assumed]
//          b  = Conv1x1(12->12, t) + body.bias + h   [residual assumed]
//   att:   a  = Sigmoid(Conv1x1(PReLU(Conv1x1(GAP(b)))))   [SE attention]
//          f  = b * a                                [per-channel scale]
//   tail:  y  = Conv3x3(3->3, PixelShuffle2(f))  ->  (1, 3, 2H, 2W)
#pragma once
#include <cuda_runtime.h>

struct LayerTimes {            // milliseconds, filled when timing enabled
    float head_block1, head_block2, head_merge;
    float body_conv3, body_conv1_res;
    float att, channel_scale;
    float pixel_shuffle, tail_conv;
    float total;
};

class SRModel {
public:
    // H, W: input spatial size. Loads weights (embedded weights.h) into
    // constant memory and allocates all activation buffers once.
    SRModel(int H, int W);
    ~SRModel();

    // d_in: device ptr (4*H*W floats), d_out: device ptr (3*2H*2W floats).
    // If times != nullptr, per-layer latency is measured with CUDA events.
    void forward(const float* d_in, float* d_out, LayerTimes* times = nullptr);

    size_t activation_bytes() const { return act_bytes_; }
    static size_t weight_bytes();

private:
    int H_, W_;
    size_t act_bytes_ = 0;
    // Activation buffers (12-channel feature maps + scratch)
    float *d_b1_ = nullptr, *d_b2_ = nullptr, *d_h_ = nullptr;
    float *d_t_ = nullptr, *d_b_ = nullptr, *d_f_ = nullptr;
    float *d_up_ = nullptr;                  // 3 x 2H x 2W pre-tail
    float *d_pool_ = nullptr, *d_att_ = nullptr;
    // Two streams: head.block1 and head.block2 are independent -> overlap.
    cudaStream_t s1_, s2_;
    cudaEvent_t  ev_b2_done_;
};
