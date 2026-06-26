"""
extract_nm_weights.py
Extract reparameterized slim weights from New Module model_best.pkl.
Produces weights_nm.bin — same format as cuda_isp/weights.bin.
"""
import sys, os, zipfile
import numpy as np
import torch

sys.path.insert(0, r'C:\Users\aujla\Desktop\archive\syenet')
from model.isp import SYEISPNet

ARCHIVE  = r'C:\Users\aujla\Desktop\archive'
NM_ZIP   = os.path.join(ARCHIVE, 'new_module.zip')
OUT_DIR  = os.path.join(ARCHIVE, 'new_module')
os.makedirs(OUT_DIR, exist_ok=True)

print("Loading New Module model_best.pkl (re-zipped) ...")
sd = torch.load(NM_ZIP, map_location='cpu', weights_only=False)
net = SYEISPNet(channels=12, rep_scale=4).eval()
net.load_state_dict(sd)
print("  Loaded OK, running .slim() reparameterization ...")
slim = net.slim().eval()
sd2  = slim.state_dict()
print("  Slim state dict keys:", len(sd2))

def w(key):
    return sd2[key].numpy().astype(np.float32)

# ── Same weight order as cuda_isp/extract_weights.py ───────────────────────────
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

weights_path = os.path.join(OUT_DIR, 'weights_nm.bin')
total = 0
with open(weights_path, 'wb') as f:
    for key, shape in order:
        arr = w(key).reshape(-1)
        f.write(arr.tobytes())
        total += len(arr)
        print(f"  {key:35s} {str(shape):20s} {len(arr):5d} floats")

print(f"\nTotal: {total} floats = {total*4} bytes  -> {weights_path}")
assert total == 5640, f"Expected 5640, got {total}"

# ── Quick sanity: run one forward pass with PyTorch slim ──────────────────────
from PIL import Image
INP_DIR = os.path.join(ARCHIVE, 'dataset', 'test', 'mediatek_raw')
GT_DIR  = os.path.join(ARCHIVE, 'dataset', 'test', 'fujifilm')

fname = sorted(f for f in os.listdir(INP_DIR) if f.endswith('.png'))[0]
print(f"\nSanity check on: {fname}")

raw = np.array(Image.open(os.path.join(INP_DIR, fname)))
h, ww = raw.shape
rggb = raw.reshape(h//2,2,ww//2,2).transpose([1,3,0,2]).reshape([-1,h//2,ww//2])
rggb = rggb.astype(np.float32) / 4095.
rggb.tofile(os.path.join(OUT_DIR, 'input_nm.bin'))

with torch.no_grad():
    inp_t = torch.from_numpy(rggb).unsqueeze(0)
    ref   = slim(inp_t).clamp(0,1).squeeze().permute(1,2,0).numpy()
ref.tofile(os.path.join(OUT_DIR, 'ref_nm.bin'))

gt = np.array(Image.open(os.path.join(GT_DIR, fname)).convert('RGB')).astype(np.float32)/255.
import math
mse  = float(((ref.astype(np.float64) - gt.astype(np.float64))**2).mean())
psnr = 10.*np.log10(1./mse)
print(f"  Slim PyTorch PSNR vs GT: {psnr:.4f} dB")
print(f"\nDone. weights_nm.bin ready for syenet_isp.exe")
