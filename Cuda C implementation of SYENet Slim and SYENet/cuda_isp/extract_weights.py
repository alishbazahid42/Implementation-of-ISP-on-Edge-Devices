"""
Extract SYEISPNetS weights from model_best.pkl (via .slim()) to weights.bin
and prepare one test image as raw float32 binary for CUDA inference.
"""
import sys, os, struct
import numpy as np
import torch
from PIL import Image

sys.path.insert(0, r'C:\Users\aujla\Desktop\archive\syenet')
from model.isp import SYEISPNet

ARCHIVE = r'C:\Users\aujla\Desktop\archive'
OUT     = r'C:\Users\aujla\Desktop\archive\cuda_isp'

# ── Load and reparameterize ───────────────────────────────────────────────────
print("Loading model_best.pkl and running .slim() ...")
net = SYEISPNet(channels=12, rep_scale=4).eval()
sd  = torch.load(os.path.join(ARCHIVE, 'model_best.pkl'), map_location='cpu', weights_only=False)
net.load_state_dict(sd)
slim = net.slim().eval()
sd2  = slim.state_dict()

def w(key):
    return sd2[key].numpy().astype(np.float32)

# ── Write weights.bin (exactly 5640 floats in defined order) ──────────────────
# This order MUST match what syenet.cu reads.
order = [
    ('head.block1.0.weight', (12,4,5,5)),   # 1200
    ('head.block1.0.bias',   (12,)),         #   12
    ('head.block1.1.weight', (12,)),         #   12  PReLU alpha
    ('head.block1.2.weight', (12,12,3,3)),  # 1296
    ('head.block1.2.bias',   (12,)),         #   12
    ('head.block2.weight',   (12,4,5,5)),   # 1200
    ('head.block2.bias',     (12,)),         #   12
    ('head.bias',            (12,)),         #   12  (squeezed from 1,12,1,1)
    ('body.block1.weight',   (12,12,3,3)),  # 1296
    ('body.block1.bias',     (12,)),         #   12
    ('body.block2.weight',   (12,12,1,1)),  #  144
    ('body.block2.bias',     (12,)),         #   12
    ('body.bias',            (12,)),         #   12
    ('att.1.weight',         (12,12,1,1)),  #  144
    ('att.1.bias',           (12,)),         #   12
    ('att.2.weight',         (12,)),         #   12  PReLU alpha
    ('att.3.weight',         (12,12,1,1)),  #  144
    ('att.3.bias',           (12,)),         #   12
    ('tail.1.weight',        (3,3,3,3)),    #   81
    ('tail.1.bias',          (3,)),          #    3
]

weights_path = os.path.join(OUT, 'weights.bin')
total = 0
with open(weights_path, 'wb') as f:
    for key, shape in order:
        arr = w(key).reshape(-1)
        f.write(arr.tobytes())
        total += len(arr)
        print(f"  {key:35s} {str(shape):20s} {len(arr):5d} floats")

print(f"\nTotal: {total} floats = {total*4} bytes  -> {weights_path}")
assert total == 5640, f"Expected 5640, got {total}"

# ── Prepare one test image ────────────────────────────────────────────────────
INP_DIR = os.path.join(ARCHIVE, 'dataset', 'test', 'mediatek_raw')
GT_DIR  = os.path.join(ARCHIVE, 'dataset', 'test', 'fujifilm')

fname = sorted(f for f in os.listdir(INP_DIR) if f.endswith('.png'))[0]
stem  = os.path.splitext(fname)[0]
print(f"\nTest image: {fname}")

raw = np.array(Image.open(os.path.join(INP_DIR, fname)))   # uint16 (H, W)
h, w_ = raw.shape
rggb = raw.reshape(h//2,2,w_//2,2).transpose([1,3,0,2]).reshape([-1,h//2,w_//2])
rggb = rggb.astype(np.float32) / 4095.                      # (4, 128, 128)
rggb.tofile(os.path.join(OUT, 'input.bin'))
print(f"Input  saved: input.bin  shape={rggb.shape}  ({rggb.nbytes} bytes)")

gt = np.array(Image.open(os.path.join(GT_DIR, fname)).convert('RGB')).astype(np.float32)/255.
gt.tofile(os.path.join(OUT, 'gt.bin'))                      # HWC (256, 256, 3)
print(f"GT     saved: gt.bin     shape={gt.shape}   ({gt.nbytes} bytes)")

# ── PyTorch reference output for PSNR verification ───────────────────────────
with torch.no_grad():
    inp_t = torch.from_numpy(rggb).unsqueeze(0)
    ref   = slim(inp_t).clamp(0,1).squeeze().permute(1,2,0).numpy()  # HWC
ref.tofile(os.path.join(OUT, 'ref.bin'))                    # HWC (256, 256, 3)
print(f"Ref    saved: ref.bin    shape={ref.shape}   ({ref.nbytes} bytes)")

# PSNR of PyTorch ref vs GT
mse  = float(((ref.astype(np.float64) - gt.astype(np.float64))**2).mean())
psnr = 10.*np.log10(1./mse)
print(f"\nPyTorch reference PSNR vs GT: {psnr:.4f} dB  (CUDA output should match this)")
print("\nDone. Ready to compile syenet.cu")
