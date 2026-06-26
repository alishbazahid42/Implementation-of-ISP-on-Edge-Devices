"""
run_cuda_benchmark.py
Run SYEISPNetS CUDA C inference on all 2417 test images.
Computes per-image PSNR, SSIM, and latency using the real CUDA exe.
Results saved to cuda_benchmark_results.json
"""
import os, sys, json, subprocess, time, tempfile
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim_fn
import math

EXE      = r"C:\Users\aujla\Desktop\archive\cuda_isp\syenet_isp.exe"
WEIGHTS  = r"C:\Users\aujla\Desktop\archive\cuda_isp\weights.bin"
RAW_DIR  = r"C:\Users\aujla\Desktop\archive\dataset\test\mediatek_raw"
GT_DIR   = r"C:\Users\aujla\Desktop\archive\dataset\test\fujifilm"
OUT_JSON = r"C:\Users\aujla\Desktop\archive\cuda_benchmark_results.json"

def psnr(a, b):
    mse = float(((a.astype(np.float64) - b.astype(np.float64))**2).mean())
    return 100.0 if mse == 0 else 10.0 * math.log10(1.0 / mse)

def raw_to_float32(png_path):
    raw = np.array(Image.open(png_path))   # uint16 H×W
    h, w = raw.shape
    rggb = raw.reshape(h//2, 2, w//2, 2).transpose([1,3,0,2]).reshape([-1, h//2, w//2])
    return (rggb.astype(np.float32) / 4095.0)  # (4,128,128)

def gt_to_float32(png_path):
    return np.array(Image.open(png_path).convert('RGB')).astype(np.float32) / 255.0  # HWC

fnames = sorted(f for f in os.listdir(RAW_DIR) if f.endswith('.png'))
total  = len(fnames)
print(f"Found {total} images. Running CUDA C inference on all {total}...")
print(f"EXE: {EXE}")

results = []
t_start = time.time()

# Use temp files for I/O
tmp_in  = os.path.join(tempfile.gettempdir(), "syenet_input.bin")
tmp_out = os.path.join(tempfile.gettempdir(), "syenet_output.bin")

for i, fname in enumerate(fnames):
    raw_path = os.path.join(RAW_DIR, fname)
    gt_path  = os.path.join(GT_DIR,  fname)

    if not os.path.exists(gt_path):
        continue

    # Convert RAW PNG → float32 binary
    rggb = raw_to_float32(raw_path)
    rggb.tofile(tmp_in)

    # Run CUDA exe (single iteration for fast per-image pass)
    t0 = time.perf_counter()
    proc = subprocess.run(
        [EXE, WEIGHTS, tmp_in, tmp_out, "1"],
        capture_output=True, text=True
    )
    t1 = time.perf_counter()

    if proc.returncode != 0:
        print(f"  ERROR on {fname}: {proc.stderr[:80]}")
        continue

    # Parse LATENCY_MS from stdout
    lat_ms = None
    for line in proc.stdout.splitlines():
        if line.startswith("LATENCY_MS="):
            lat_ms = float(line.split("=")[1])
            break

    # Load output (CHW → HWC)
    out_chw = np.fromfile(tmp_out, dtype=np.float32).reshape(3, 256, 256)
    out_hwc = out_chw.transpose(1, 2, 0).clip(0, 1)

    # Load GT
    gt_hwc = gt_to_float32(gt_path)

    # Compute PSNR and SSIM
    img_psnr = psnr(out_hwc, gt_hwc)
    img_ssim = float(ssim_fn(out_hwc, gt_hwc,
                              data_range=1.0, channel_axis=2,
                              win_size=7))

    results.append({
        "file":    fname,
        "psnr":    round(img_psnr, 4),
        "ssim":    round(img_ssim, 6),
        "lat_ms":  round(lat_ms, 4) if lat_ms else None,
        "wall_ms": round((t1 - t0) * 1000, 1),
    })

    if (i + 1) % 100 == 0 or i == 0:
        elapsed = time.time() - t_start
        rate    = (i + 1) / elapsed
        eta     = (total - i - 1) / rate
        avg_psnr = sum(r["psnr"] for r in results) / len(results)
        avg_ssim = sum(r["ssim"] for r in results) / len(results)
        print(f"  [{i+1:4d}/{total}]  PSNR={avg_psnr:.3f}  SSIM={avg_ssim:.4f}"
              f"  rate={rate:.1f} img/s  ETA={eta:.0f}s")

elapsed_total = time.time() - t_start

# ── Summary stats ─────────────────────────────────────────────────────────────
psnrs    = [r["psnr"]    for r in results]
ssims    = [r["ssim"]    for r in results]
lats     = [r["lat_ms"]  for r in results if r["lat_ms"] is not None]
walls    = [r["wall_ms"] for r in results]

summary = {
    "total_images":   len(results),
    "elapsed_sec":    round(elapsed_total, 1),
    "psnr": {
        "mean":   round(float(np.mean(psnrs)),  4),
        "min":    round(float(np.min(psnrs)),   4),
        "max":    round(float(np.max(psnrs)),   4),
        "std":    round(float(np.std(psnrs)),   4),
        "p5":     round(float(np.percentile(psnrs,  5)), 4),
        "p25":    round(float(np.percentile(psnrs, 25)), 4),
        "median": round(float(np.median(psnrs)),         4),
        "p75":    round(float(np.percentile(psnrs, 75)), 4),
        "p95":    round(float(np.percentile(psnrs, 95)), 4),
    },
    "ssim": {
        "mean":   round(float(np.mean(ssims)),  6),
        "min":    round(float(np.min(ssims)),   6),
        "max":    round(float(np.max(ssims)),   6),
        "std":    round(float(np.std(ssims)),   6),
        "p5":     round(float(np.percentile(ssims,  5)), 6),
        "median": round(float(np.median(ssims)),         6),
        "p95":    round(float(np.percentile(ssims, 95)), 6),
    },
    "gpu_latency_ms": {
        "mean":   round(float(np.mean(lats)),  4),
        "min":    round(float(np.min(lats)),   4),
        "max":    round(float(np.max(lats)),   4),
        "std":    round(float(np.std(lats)),   4),
        "median": round(float(np.median(lats)), 4),
    },
    "fps_single_pass": round(1000.0 / float(np.mean(lats)), 1),
}

# Save
data = {"summary": summary, "per_image": results}
with open(OUT_JSON, "w") as f:
    json.dump(data, f, indent=2)

print()
print("=" * 60)
print(f"  Total images:   {summary['total_images']}")
print(f"  PSNR mean/std:  {summary['psnr']['mean']:.3f} ± {summary['psnr']['std']:.3f} dB")
print(f"  SSIM mean/std:  {summary['ssim']['mean']:.4f} ± {summary['ssim']['std']:.4f}")
print(f"  GPU latency:    {summary['gpu_latency_ms']['mean']:.3f} ms avg")
print(f"  FPS (1-pass):   {summary['fps_single_pass']:.1f}")
print(f"  Elapsed:        {elapsed_total:.1f} s")
print(f"  Results saved:  {OUT_JSON}")
print("=" * 60)
