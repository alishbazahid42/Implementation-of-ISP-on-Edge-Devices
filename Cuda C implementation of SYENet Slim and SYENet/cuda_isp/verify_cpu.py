"""
verify_cpu.py  —  compare CPU C output vs GPU CUDA output and PyTorch reference.
"""
import numpy as np, math, os

DIR = os.path.dirname(os.path.abspath(__file__))

def psnr(a, b):
    mse = float(((a.astype(np.float64) - b.astype(np.float64))**2).mean())
    return 100. if mse == 0 else 10.*math.log10(1./mse)

def load(name, shape):
    path = os.path.join(DIR, name)
    if not os.path.exists(path):
        print(f"  MISSING: {name}")
        return None
    arr = np.fromfile(path, dtype=np.float32).reshape(shape)
    return arr

print("=" * 50)
print("  SYEISPNetS  CPU vs GPU verification")
print("=" * 50)

cpu = load('output_cpu.bin', (3, 256, 256))
gpu = load('output.bin',     (3, 256, 256))
ref = load('ref.bin',        (256, 256, 3))
gt  = load('gt.bin',         (256, 256, 3))

if cpu is None:
    print("Run syenet_cpu.exe first.")
    import sys; sys.exit(1)

# Convert CHW → HWC for comparison
cpu_hwc = cpu.transpose(1, 2, 0)
gpu_hwc = gpu.transpose(1, 2, 0) if gpu is not None else None

print(f"\nCPU output range:  [{cpu_hwc.min():.4f}, {cpu_hwc.max():.4f}]")
if gpu is not None:
    print(f"GPU output range:  [{gpu_hwc.min():.4f}, {gpu_hwc.max():.4f}]")

if gpu is not None:
    p_cg  = psnr(cpu_hwc, gpu_hwc)
    diff  = float(np.abs(cpu_hwc.astype(np.float64) - gpu_hwc.astype(np.float64)).max())
    print(f"\nCPU vs GPU:")
    print(f"  PSNR:    {p_cg:.4f} dB")
    print(f"  Max diff:{diff:.6f}")
    if p_cg > 60:
        print("  Verdict: PERFECT — bit-identical")
    elif p_cg > 45:
        print("  Verdict: EXCELLENT — minor rounding differences")
    else:
        print("  Verdict: MISMATCH — check CPU kernel logic")

if ref is not None:
    p_cr = psnr(cpu_hwc, ref)
    print(f"\nCPU vs PyTorch ref:  PSNR = {p_cr:.4f} dB")

if gt is not None:
    p_cgt = psnr(cpu_hwc, gt)
    print(f"CPU vs Fujifilm GT:  PSNR = {p_cgt:.4f} dB  (expected ~24.83 dB avg)")

try:
    from PIL import Image
    comp = np.ones((256, 256*3 + 8, 3), dtype=np.uint8) * 180
    comp[:, :256]     = (cpu_hwc.clip(0,1)*255).astype(np.uint8)
    if gpu is not None:
        comp[:, 260:516]  = (gpu_hwc.clip(0,1)*255).astype(np.uint8)
    if gt is not None:
        comp[:, 520:]     = (gt.clip(0,1)*255).astype(np.uint8)
    Image.fromarray(comp).save(os.path.join(DIR, 'comparison_cpu.png'))
    print(f"\nSaved: comparison_cpu.png  (CPU | GPU | GT)")
except ImportError:
    pass

print("\n" + "=" * 50)
