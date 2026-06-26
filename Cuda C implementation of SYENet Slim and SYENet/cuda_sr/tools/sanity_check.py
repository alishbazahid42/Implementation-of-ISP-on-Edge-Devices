"""Pure-numpy forward of the recovered net: range stats + INT8 analysis."""
import numpy as np, json

meta = json.load(open('state_dict_meta.json'))
T = {k: np.fromfile('data/' + v['storage_key'], dtype='<f4').reshape(v['shape'])
     for k, v in meta.items()}

print("Per-tensor weight ranges (INT8 quantization relevance):")
for k, w in T.items():
    print(f"  {k:25s} min={w.min():9.3f} max={w.max():8.3f} absmean={np.abs(w).mean():7.4f}")

def conv(x, w, b, pad):
    Co, Ci, K, _ = w.shape
    H, W = x.shape[1:]
    xp = np.pad(x, ((0, 0), (pad, pad), (pad, pad)))
    out = np.zeros((Co, H, W), np.float32)
    for co in range(Co):
        acc = np.full((H, W), b[co], np.float32)
        for ci in range(Ci):
            for ky in range(K):
                for kx in range(K):
                    acc += w[co, ci, ky, kx] * xp[ci, ky:ky+H, kx:kx+W]
        out[co] = acc
    return out

def prelu(x, a): return np.where(x >= 0, x, x * a[:, None, None])

np.random.seed(42)
H = W = 32
x = np.random.rand(4, H, W).astype(np.float32)

b1 = conv(x, T['head.block1.0.weight'], T['head.block1.0.bias'], 2)
b1 = prelu(b1, T['head.block1.1.weight'])
b1 = conv(b1, T['head.block1.2.weight'], T['head.block1.2.bias'], 1)
b2 = conv(x, T['head.block2.weight'], T['head.block2.bias'], 2)
h = b1 + b2 + T['head.bias'][0]

t = np.maximum(conv(h, T['body.block1.weight'], T['body.block1.bias'], 1), 0)
b = conv(t, T['body.block2.weight'], T['body.block2.bias'], 0) + T['body.bias'][0] + h

g = b.mean(axis=(1, 2))
hid = T['att.1.weight'][:, :, 0, 0] @ g + T['att.1.bias']
hid = np.where(hid >= 0, hid, hid * T['att.2.weight'])
a = 1 / (1 + np.exp(-(T['att.3.weight'][:, :, 0, 0] @ hid + T['att.3.bias'])))
f = b * a[:, None, None]

up = np.zeros((3, 2*H, 2*W), np.float32)
for c in range(3):
    for dy in range(2):
        for dx in range(2):
            up[c, dy::2, dx::2] = f[c*4 + dy*2 + dx]
y = conv(up, T['tail.1.weight'], T['tail.1.bias'], 1)

print(f"\nForward sanity (random [0,1] input {H}x{W}):")
print(f"  attention vector: {np.round(a, 3)}")
print(f"  output range: [{y.min():.3f}, {y.max():.3f}]  mean={y.mean():.3f}")
print("  (plausible if roughly in/near [0,1] for image data)")
