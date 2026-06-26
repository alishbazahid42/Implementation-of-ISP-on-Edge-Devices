/*
 * syenet_cpu.c  —  Plain C CPU inference for SYEISPNetS (slim model)
 *
 * No CUDA, no libraries beyond math.h.
 * Reads the same weights.bin / input.bin as the GPU version.
 * Writes output_cpu.bin (CHW float32, same format as GPU output).
 *
 * Compile (MSVC):  cl /O2 /fp:fast /arch:AVX2 syenet_cpu.c /Fe:syenet_cpu.exe /link /SUBSYSTEM:CONSOLE
 * Run:             syenet_cpu.exe weights.bin input.bin output_cpu.bin
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
static double ms_now(void) {
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart / (double)f.QuadPart * 1000.0;
}
#else
#include <time.h>
static double ms_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}
#endif

/* ── Dimensions ────────────────────────────────────────────────────────────── */
#define H_IN   128
#define W_IN   128
#define C_IN   4
#define C_MID  12
#define C_OUT  3
#define H_OUT  256
#define W_OUT  256

#define TOTAL_WEIGHTS 5640

/* ── Weight offsets (identical to syenet.cu) ───────────────────────────────── */
#define OFF_H_B10_W    0
#define OFF_H_B10_B 1200
#define OFF_H_B11_A 1212
#define OFF_H_B12_W 1224
#define OFF_H_B12_B 2520
#define OFF_H_B2_W  2532
#define OFF_H_B2_B  3732
#define OFF_H_BIAS  3744
#define OFF_BO1_W   3756
#define OFF_BO1_B   5052
#define OFF_BO2_W   5064
#define OFF_BO2_B   5208
#define OFF_BO_BIAS 5220
#define OFF_AT1_W   5232
#define OFF_AT1_B   5376
#define OFF_AT2_A   5388
#define OFF_AT3_W   5400
#define OFF_AT3_B   5544
#define OFF_TL1_W   5556
#define OFF_TL1_B   5637

static float W_all[TOTAL_WEIGHTS];

/* ══════════════════════════════════════════════════════════════════════════ *
 *  CPU OPERATIONS
 * ══════════════════════════════════════════════════════════════════════════ */

/*
 * conv2d  —  out[co,oh,ow] = sum_{ci,kh,kw} in[ci,oh+kh-pad,ow+kw-pad]*w[co,ci,kh,kw] + b[co]
 */
static void conv2d(
    const float *in, float *out,
    int w_off, int b_off,
    int C_in, int C_out,
    int H, int W,
    int KH, int KW,
    int pad_h, int pad_w)
{
    for (int co = 0; co < C_out; co++) {
        float bias = W_all[b_off + co];
        for (int oh = 0; oh < H; oh++) {
            for (int ow = 0; ow < W; ow++) {
                float acc = bias;
                for (int ci = 0; ci < C_in; ci++) {
                    int w_base = w_off + co*(C_in*KH*KW) + ci*(KH*KW);
                    for (int kh = 0; kh < KH; kh++) {
                        int ih = oh + kh - pad_h;
                        if (ih < 0 || ih >= H) continue;
                        for (int kw = 0; kw < KW; kw++) {
                            int iw = ow + kw - pad_w;
                            if (iw < 0 || iw >= W) continue;
                            acc += in[ci*H*W + ih*W + iw]
                                 * W_all[w_base + kh*KW + kw];
                        }
                    }
                }
                out[co*H*W + oh*W + ow] = acc;
            }
        }
    }
}

/* prelu  —  per-channel: x >= 0 ? x : alpha[c]*x  (tensor: C,HW) */
static void prelu(float *data, int alpha_off, int C, int HW) {
    for (int c = 0; c < C; c++) {
        float alpha = W_all[alpha_off + c];
        for (int i = 0; i < HW; i++) {
            float v = data[c*HW + i];
            data[c*HW + i] = v >= 0.f ? v : alpha * v;
        }
    }
}

/* mul_bias  —  out[c,hw] = a[c,hw] * b[c,hw] + bias[c] */
static void mul_bias(const float *a, const float *b, float *out,
                     int bias_off, int C, int HW) {
    for (int c = 0; c < C; c++) {
        float bias = W_all[bias_off + c];
        for (int i = 0; i < HW; i++) {
            int idx = c*HW + i;
            out[idx] = a[idx] * b[idx] + bias;
        }
    }
}

/* gap  —  (C,H,W) → (C,) global average */
static void gap(const float *in, float *g, int C, int HW) {
    for (int c = 0; c < C; c++) {
        float s = 0.f;
        for (int i = 0; i < HW; i++) s += in[c*HW + i];
        g[c] = s / (float)HW;
    }
}

/* fc1x1  —  out[co] = sum_ci in[ci] * w[co,ci] + b[co]  (SE attention FC) */
static void fc1x1(const float *in, float *out,
                  int w_off, int b_off, int C_in, int C_out) {
    for (int co = 0; co < C_out; co++) {
        float acc = W_all[b_off + co];
        for (int ci = 0; ci < C_in; ci++)
            acc += in[ci] * W_all[w_off + co*C_in + ci];
        out[co] = acc;
    }
}

/* sigmoid_ip  —  in-place 1/(1+exp(-x)) */
static void sigmoid_ip(float *data, int N) {
    for (int i = 0; i < N; i++)
        data[i] = 1.f / (1.f + expf(-data[i]));
}

/* channel_scale  —  x[c,hw] *= att[c]  (in-place) */
static void channel_scale(float *data, const float *att, int C, int HW) {
    for (int c = 0; c < C; c++) {
        float a = att[c];
        for (int i = 0; i < HW; i++)
            data[c*HW + i] *= a;
    }
}

/* pixel_shuffle  —  (C*4,H,W) → (C,2H,2W) with scale=2
   out[co, oh, ow] = in[ co*4 + (oh%2)*2 + (ow%2),  oh/2,  ow/2 ] */
static void pixel_shuffle(const float *in, float *out,
                           int C_out, int H_out, int W_out) {
    int H_in = H_out >> 1, W_in = W_out >> 1;
    for (int co = 0; co < C_out; co++) {
        for (int oh = 0; oh < H_out; oh++) {
            int h_in = oh >> 1;
            for (int ow = 0; ow < W_out; ow++) {
                int c_in = co*4 + (oh & 1)*2 + (ow & 1);
                int w_in = ow >> 1;
                out[co*H_out*W_out + oh*W_out + ow] =
                    in[c_in*H_in*W_in + h_in*W_in + w_in];
            }
        }
    }
}

/* clamp01  —  in-place clamp to [0, 1] */
static void clamp01(float *data, int N) {
    for (int i = 0; i < N; i++) {
        float v = data[i];
        data[i] = v < 0.f ? 0.f : (v > 1.f ? 1.f : v);
    }
}

/* ══════════════════════════════════════════════════════════════════════════ *
 *  FORWARD PASS
 * ══════════════════════════════════════════════════════════════════════════ */
static void forward(
    const float *x0,  /* (C_IN,  H_IN, W_IN)   input  */
    float *b1,        /* (C_MID, H_IN, W_IN)   work   */
    float *b2,        /* (C_MID, H_IN, W_IN)   work   */
    float *x1,        /* (C_MID, H_IN, W_IN)   work   */
    float *gv,        /* (C_MID,)              gap    */
    float *ps,        /* (C_OUT, H_OUT, W_OUT) work   */
    float *out)       /* (C_OUT, H_OUT, W_OUT) output */
{
    const int H = H_IN, W = W_IN, C = C_MID, HW = H*W;

    /* ── HEAD ──────────────────────────────────────────────────────────── */
    conv2d(x0, b1, OFF_H_B10_W, OFF_H_B10_B, C_IN, C, H, W, 5, 5, 2, 2);
    prelu(b1, OFF_H_B11_A, C, HW);
    conv2d(b1, x1, OFF_H_B12_W, OFF_H_B12_B, C, C, H, W, 3, 3, 1, 1);
    conv2d(x0, b2, OFF_H_B2_W,  OFF_H_B2_B,  C_IN, C, H, W, 5, 5, 2, 2);
    mul_bias(x1, b2, b1, OFF_H_BIAS, C, HW);

    /* ── BODY ──────────────────────────────────────────────────────────── */
    conv2d(b1, x1, OFF_BO1_W, OFF_BO1_B, C, C, H, W, 3, 3, 1, 1);
    conv2d(b1, b2, OFF_BO2_W, OFF_BO2_B, C, C, H, W, 1, 1, 0, 0);
    mul_bias(x1, b2, b1, OFF_BO_BIAS, C, HW);

    /* ── ATT (SE) ───────────────────────────────────────────────────────── */
    gap(b1, gv, C, HW);
    fc1x1(gv, b2, OFF_AT1_W, OFF_AT1_B, C, C);    /* reuse b2 as temp (C elems) */
    prelu(b2, OFF_AT2_A, C, 1);                    /* HW=1 for vector */
    fc1x1(b2, gv, OFF_AT3_W, OFF_AT3_B, C, C);
    sigmoid_ip(gv, C);
    channel_scale(b1, gv, C, HW);

    /* ── TAIL ───────────────────────────────────────────────────────────── */
    pixel_shuffle(b1, ps, C_OUT, H_OUT, W_OUT);
    conv2d(ps, out, OFF_TL1_W, OFF_TL1_B, C_OUT, C_OUT, H_OUT, W_OUT, 3, 3, 1, 1);
    clamp01(out, C_OUT*H_OUT*W_OUT);
}

/* ══════════════════════════════════════════════════════════════════════════ *
 *  MAIN
 * ══════════════════════════════════════════════════════════════════════════ */
int main(int argc, char **argv)
{
    const char *weights_path = argc > 1 ? argv[1] : "weights.bin";
    const char *input_path   = argc > 2 ? argv[2] : "input.bin";
    const char *output_path  = argc > 3 ? argv[3] : "output_cpu.bin";

    printf("SYEISPNetS CPU inference\n");
    printf("========================\n");

    /* Load weights */
    printf("Loading weights: %s\n", weights_path);
    FILE *f = fopen(weights_path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", weights_path); return 1; }
    size_t nr = fread(W_all, sizeof(float), TOTAL_WEIGHTS, f);
    fclose(f);
    if ((int)nr != TOTAL_WEIGHTS) {
        fprintf(stderr, "Expected %d floats, got %zu\n", TOTAL_WEIGHTS, nr); return 1;
    }
    printf("  %d weights (%.1f KB) loaded\n", TOTAL_WEIGHTS, TOTAL_WEIGHTS*4.f/1024.f);

    /* Load input */
    int in_size = C_IN * H_IN * W_IN;
    float *h_in = (float *)malloc(in_size * sizeof(float));
    f = fopen(input_path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", input_path); return 1; }
    fread(h_in, sizeof(float), in_size, f);
    fclose(f);
    printf("Loading input:   %s  (%d × %d × %d)\n", input_path, C_IN, H_IN, W_IN);

    /* Allocate buffers */
    float *b1  = (float *)malloc(C_MID * H_IN  * W_IN  * sizeof(float));
    float *b2  = (float *)malloc(C_MID * H_IN  * W_IN  * sizeof(float));
    float *x1  = (float *)malloc(C_MID * H_IN  * W_IN  * sizeof(float));
    float *gv  = (float *)malloc(C_MID          * sizeof(float));
    float *ps  = (float *)malloc(C_OUT * H_OUT * W_OUT * sizeof(float));
    float *out = (float *)malloc(C_OUT * H_OUT * W_OUT * sizeof(float));

    /* Warmup */
    printf("Warmup (5 runs)...\n");
    for (int i = 0; i < 5; i++)
        forward(h_in, b1, b2, x1, gv, ps, out);

    /* Timed benchmark */
    const int N_ITER = 50;
    double times[50];
    double total = 0.0;
    double t_min = 1e18, t_max = 0.0;

    printf("Timed run (%d iterations)...\n", N_ITER);
    for (int i = 0; i < N_ITER; i++) {
        double t0 = ms_now();
        forward(h_in, b1, b2, x1, gv, ps, out);
        double t1 = ms_now();
        times[i] = t1 - t0;
        total += times[i];
        if (times[i] < t_min) t_min = times[i];
        if (times[i] > t_max) t_max = times[i];
    }

    double avg = total / N_ITER;

    /* Median */
    /* simple insertion sort for 50 elements */
    double sorted[50];
    memcpy(sorted, times, N_ITER * sizeof(double));
    for (int i = 1; i < N_ITER; i++) {
        double key = sorted[i];
        int j = i - 1;
        while (j >= 0 && sorted[j] > key) { sorted[j+1] = sorted[j]; j--; }
        sorted[j+1] = key;
    }
    double median = (N_ITER % 2 == 0)
        ? (sorted[N_ITER/2-1] + sorted[N_ITER/2]) / 2.0
        : sorted[N_ITER/2];

    /* Std dev */
    double var = 0.0;
    for (int i = 0; i < N_ITER; i++) var += (times[i]-avg)*(times[i]-avg);
    double stdev = sqrt(var / N_ITER);

    printf("\n");
    printf("===========================================\n");
    printf("  SYEISPNetS  CPU inference results\n");
    printf("===========================================\n");
    printf("  Iterations : %d\n",       N_ITER);
    printf("  Avg latency: %.3f ms\n",  avg);
    printf("  Min latency: %.3f ms\n",  t_min);
    printf("  Max latency: %.3f ms\n",  t_max);
    printf("  Median     : %.3f ms\n",  median);
    printf("  Std dev    : %.3f ms\n",  stdev);
    printf("  Throughput : %.2f FPS\n", 1000.0 / avg);
    printf("===========================================\n");

    /* Save output */
    int out_size = C_OUT * H_OUT * W_OUT;
    f = fopen(output_path, "wb");
    if (!f) { fprintf(stderr, "Cannot write %s\n", output_path); return 1; }
    fwrite(out, sizeof(float), out_size, f);
    fclose(f);
    printf("Output saved: %s  (CHW: %d x %d x %d)\n",
           output_path, C_OUT, H_OUT, W_OUT);

    free(h_in); free(b1); free(b2); free(x1); free(gv); free(ps); free(out);
    printf("Run verify_cpu.py to check PSNR vs GPU output.\n");
    return 0;
}
