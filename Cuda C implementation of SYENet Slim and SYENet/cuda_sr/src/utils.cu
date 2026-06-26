#include "utils.h"
#include "weights.h"
#include <cmath>
#include <random>
#include <cstring>

// ---------------------------------------------------------------------------
// CPU reference. Mirrors model.cu exactly (same op order, same assumptions),
// but accumulates in double so it doubles as a numerical ground truth.
// ---------------------------------------------------------------------------
static void conv2d_cpu(const float* in, float* out, int Cin, int Cout,
                       int H, int W, int K, int w_off, int b_off,
                       int act /*0 none,1 relu,2 prelu*/, int prelu_off) {
    const int P = K / 2;
    for (int oc = 0; oc < Cout; ++oc)
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x) {
                double acc = MODEL_WEIGHTS[b_off + oc];
                for (int ic = 0; ic < Cin; ++ic)
                    for (int ky = 0; ky < K; ++ky)
                        for (int kx = 0; kx < K; ++kx) {
                            int iy = y + ky - P, ix = x + kx - P;
                            if (iy < 0 || iy >= H || ix < 0 || ix >= W) continue;
                            acc += (double)in[((size_t)ic * H + iy) * W + ix] *
                                   MODEL_WEIGHTS[w_off + ((oc * Cin + ic) * K + ky) * K + kx];
                        }
                float v = (float)acc;
                if (act == 1) v = v > 0 ? v : 0;
                else if (act == 2) v = v >= 0 ? v : v * MODEL_WEIGHTS[prelu_off + oc];
                out[((size_t)oc * H + y) * W + x] = v;
            }
}

void cpu_forward(const float* x, float* y, int H, int W) {
    const int C = 12;
    const size_t pl = (size_t)H * W;
    std::vector<float> b1(C * pl), b2(C * pl), h(C * pl), t(C * pl), b(C * pl),
                       up(3 * 4 * pl);

    // head
    conv2d_cpu(x, t.data(), 4, C, H, W, 5, HEAD_BLOCK1_0_WEIGHT_OFFSET,
               HEAD_BLOCK1_0_BIAS_OFFSET, 2, HEAD_BLOCK1_1_PRELU_OFFSET);
    conv2d_cpu(t.data(), b1.data(), C, C, H, W, 3, HEAD_BLOCK1_2_WEIGHT_OFFSET,
               HEAD_BLOCK1_2_BIAS_OFFSET, 0, 0);
    conv2d_cpu(x, b2.data(), 4, C, H, W, 5, HEAD_BLOCK2_WEIGHT_OFFSET,
               HEAD_BLOCK2_BIAS_OFFSET, 0, 0);
    for (int c = 0; c < C; ++c)
        for (size_t i = 0; i < pl; ++i)
            h[c * pl + i] = b1[c * pl + i] + b2[c * pl + i] +
                            MODEL_WEIGHTS[HEAD_BIAS_OFFSET + c];

    // body (+ residual)
    conv2d_cpu(h.data(), t.data(), C, C, H, W, 3, BODY_BLOCK1_WEIGHT_OFFSET,
               BODY_BLOCK1_BIAS_OFFSET, 1, 0);
    conv2d_cpu(t.data(), b.data(), C, C, H, W, 1, BODY_BLOCK2_WEIGHT_OFFSET,
               BODY_BLOCK2_BIAS_OFFSET, 0, 0);
    for (int c = 0; c < C; ++c)
        for (size_t i = 0; i < pl; ++i)
            b[c * pl + i] += h[c * pl + i] + MODEL_WEIGHTS[BODY_BIAS_OFFSET + c];

    // SE attention
    double pooled[12], hid[12], att[12];
    for (int c = 0; c < C; ++c) {
        double s = 0;
        for (size_t i = 0; i < pl; ++i) s += b[c * pl + i];
        pooled[c] = s / (double)pl;
    }
    for (int c = 0; c < C; ++c) {
        double v = MODEL_WEIGHTS[ATT_1_BIAS_OFFSET + c];
        for (int i = 0; i < C; ++i) v += MODEL_WEIGHTS[ATT_1_WEIGHT_OFFSET + c * C + i] * pooled[i];
        hid[c] = v >= 0 ? v : v * MODEL_WEIGHTS[ATT_2_PRELU_OFFSET + c];
    }
    for (int c = 0; c < C; ++c) {
        double v = MODEL_WEIGHTS[ATT_3_BIAS_OFFSET + c];
        for (int i = 0; i < C; ++i) v += MODEL_WEIGHTS[ATT_3_WEIGHT_OFFSET + c * C + i] * hid[i];
        att[c] = 1.0 / (1.0 + std::exp(-v));
    }
    for (int c = 0; c < C; ++c)
        for (size_t i = 0; i < pl; ++i) b[c * pl + i] *= (float)att[c];

    // tail: PixelShuffle(2) then conv3x3 3->3
    const int Ho = 2 * H, Wo = 2 * W;
    for (int c = 0; c < 3; ++c)
        for (int yy = 0; yy < Ho; ++yy)
            for (int xx = 0; xx < Wo; ++xx) {
                int ci = c * 4 + (yy & 1) * 2 + (xx & 1);
                up[((size_t)c * Ho + yy) * Wo + xx] =
                    b[((size_t)ci * H + (yy >> 1)) * W + (xx >> 1)];
            }
    conv2d_cpu(up.data(), y, 3, 3, Ho, Wo, 3, TAIL_1_WEIGHT_OFFSET,
               TAIL_1_BIAS_OFFSET, 0, 0);
}

// ---------------------------------------------------------------------------
Metrics compare(const float* ref, const float* test, size_t n, float peak) {
    double se = 0, ae = 0, mx = 0;
    for (size_t i = 0; i < n; ++i) {
        double d = (double)ref[i] - test[i];
        se += d * d; ae += std::fabs(d); mx = std::max(mx, std::fabs(d));
    }
    Metrics m;
    m.mse = se / n;
    m.mae = ae / n;
    m.max_abs_err = mx;
    m.psnr = m.mse > 0 ? 10.0 * std::log10((double)peak * peak / m.mse) : 999.0;
    return m;
}

void fill_random(std::vector<float>& v, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> d(0.0f, 1.0f);
    for (auto& x : v) x = d(rng);
}

bool read_bin(const char* path, std::vector<float>& v, size_t expect) {
    FILE* f = fopen(path, "rb");
    if (!f) return false;
    v.resize(expect);
    size_t got = fread(v.data(), sizeof(float), expect, f);
    fclose(f);
    return got == expect;
}

bool write_bin(const char* path, const float* p, size_t n) {
    FILE* f = fopen(path, "wb");
    if (!f) return false;
    fwrite(p, sizeof(float), n, f);
    fclose(f);
    return true;
}
