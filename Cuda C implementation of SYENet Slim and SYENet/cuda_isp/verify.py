"""
verify.py  —  Compare CUDA output vs PyTorch reference and GT image.

Expected files (all created by extract_weights.py):
  input.bin   float32 CHW (4, 128, 128)   RGGB input
  ref.bin     float32 HWC (256, 256, 3)   PyTorch slim output
  gt.bin      float32 HWC (256, 256, 3)   Fujifilm ground-truth
  output.bin  float32 CHW (3, 256, 256)   CUDA program output

Run after: syenet_isp.exe weights.bin input.bin output.bin
"""
import numpy as np
import math
import os
import sys

DIR = os.path.dirname(os.path.abspath(__file__))

def psnr(a, b):
    mse = float(((a.astype(np.float64) - b.astype(np.float64))**2).mean())
    return 100. if mse == 0 else 10. * math.log10(1. / mse)

def mae(a, b):
    return float(np.abs(a.astype(np.float64) - b.astype(np.float64)).mean())

def load(path, shape):
    arr = np.fromfile(path, dtype=np.float32)
    try:
        return arr.reshape(shape)
    except Exception:
        print(f"  ERROR: {path} has {arr.size} floats, expected {np.prod(shape)}")
        return None

print("=" * 52)
print("  SYEISPNetS CUDA verification")
print("=" * 52)

# Load all files
output_chw = load(os.path.join(DIR, 'output.bin'), (3, 256, 256))
ref_hwc    = load(os.path.join(DIR, 'ref.bin'),    (256, 256, 3))
gt_hwc     = load(os.path.join(DIR, 'gt.bin'),     (256, 256, 3))

if output_chw is None:
    print("\noutput.bin not found or wrong size.")
    print("Run: syenet_isp.exe weights.bin input.bin output.bin")
    sys.exit(1)

# Convert CUDA output CHW → HWC to match reference
output_hwc = output_chw.transpose(1, 2, 0)  # (256, 256, 3)

print(f"\nCUDA output range:     [{output_hwc.min():.4f}, {output_hwc.max():.4f}]")
print(f"PyTorch ref range:     [{ref_hwc.min():.4f}, {ref_hwc.max():.4f}]")
print(f"GT range:              [{gt_hwc.min():.4f}, {gt_hwc.max():.4f}]")

# ── CUDA vs PyTorch reference ──────────────────────────────────────────────
p_ref  = psnr(output_hwc, ref_hwc)
m_ref  = mae(output_hwc, ref_hwc)
max_diff = float(np.abs(output_hwc.astype(np.float64)
                       - ref_hwc.astype(np.float64)).max())

print()
print(f"CUDA vs PyTorch reference:")
print(f"  PSNR:    {p_ref:.4f} dB")
print(f"  MAE:     {m_ref:.6f}")
print(f"  Max diff:{max_diff:.6f}")

# Interpretation
if p_ref > 60:
    verdict_ref = "PERFECT  (bit-identical within fp32 rounding)"
elif p_ref > 45:
    verdict_ref = "EXCELLENT  (minor fp32 rounding differences)"
elif p_ref > 35:
    verdict_ref = "GOOD  (small numerical differences)"
else:
    verdict_ref = "MISMATCH  — check kernel logic"
print(f"  Verdict: {verdict_ref}")

# ── CUDA vs ground truth ───────────────────────────────────────────────────
p_gt = psnr(output_hwc, gt_hwc)
print()
print(f"CUDA vs Fujifilm GT:")
print(f"  PSNR:    {p_gt:.4f} dB  (expected ~24.83 dB)")

if ref_hwc is not None:
    p_ref_gt = psnr(ref_hwc, gt_hwc)
    print(f"  PyTorch PSNR vs GT: {p_ref_gt:.4f} dB")
    diff_from_pytorch = abs(p_gt - p_ref_gt)
    print(f"  CUDA vs PyTorch delta: {diff_from_pytorch:.4f} dB")

print()
print("=" * 52)

# ── Optional: save side-by-side comparison PNG ────────────────────────────
try:
    from PIL import Image
    cuda_img = (output_hwc.clip(0, 1) * 255).astype(np.uint8)
    ref_img  = (ref_hwc.clip(0, 1)   * 255).astype(np.uint8)
    gt_img   = (gt_hwc.clip(0, 1)    * 255).astype(np.uint8)

    W, H = 256, 256
    gap_px = 4
    composite = np.ones((H, W*3 + gap_px*2, 3), dtype=np.uint8) * 128
    composite[:, :W]                   = cuda_img
    composite[:, W+gap_px:2*W+gap_px]  = ref_img
    composite[:, 2*W+2*gap_px:]        = gt_img

    out_png = os.path.join(DIR, 'comparison.png')
    Image.fromarray(composite).save(out_png)
    print(f"Saved side-by-side: comparison.png")
    print(f"  Left=CUDA  |  Middle=PyTorch ref  |  Right=GT")
except ImportError:
    print("(pip install Pillow to save comparison.png)")

print()
if p_ref > 45:
    print("SUCCESS: CUDA inference matches PyTorch.")
else:
    print("WARNING: Large difference from PyTorch reference.")
    print("Check kernel constants and weight offsets in syenet.cu.")
