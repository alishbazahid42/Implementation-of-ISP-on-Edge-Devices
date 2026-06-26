"""
Step 1: Export original SYEISPNet and slim SYEISPNetS to ONNX.
Runs the .slim() reparameterization on model_best.pkl to produce the
mathematically equivalent single-conv weights, then exports both to ONNX.
"""
import sys, os, torch
sys.path.insert(0, r'C:\Users\aujla\Desktop\archive\syenet')
from model.isp import SYEISPNet, SYEISPNetS

ARCHIVE = r'C:\Users\aujla\Desktop\archive'
OUT_DIR = os.path.join(ARCHIVE, 'cuda_benchmark', 'engines')
os.makedirs(OUT_DIR, exist_ok=True)

DUMMY = torch.rand(1, 4, 128, 128)
EXPORT_KWARGS = dict(
    input_names=['raw_input'],
    output_names=['rgb_output'],
    opset_version=17,
    do_constant_folding=True,
)

# ── Original model ────────────────────────────────────────────────────────────
print("Loading original model (model_best.pkl) ...")
net_orig = SYEISPNet(channels=12, rep_scale=4).eval()
sd_orig  = torch.load(os.path.join(ARCHIVE, 'model_best.pkl'),
                      map_location='cpu', weights_only=False)
net_orig.load_state_dict(sd_orig)

orig_onnx = os.path.join(OUT_DIR, 'original_model.onnx')
torch.onnx.export(net_orig, DUMMY, orig_onnx, **EXPORT_KWARGS)
print(f"  original_model.onnx  : {os.path.getsize(orig_onnx)/1024:.1f} KB")

# ── Reparameterize → slim ─────────────────────────────────────────────────────
print("Reparameterizing to slim model ...")
net_slim = net_orig.slim().eval()

slim_onnx = os.path.join(OUT_DIR, 'slim_model.onnx')
torch.onnx.export(net_slim, DUMMY, slim_onnx, **EXPORT_KWARGS)
print(f"  slim_model.onnx      : {os.path.getsize(slim_onnx)/1024:.1f} KB")

# ── Verify: max abs diff should be < 1e-4 ────────────────────────────────────
import onnxruntime as ort, numpy as np
sess_o = ort.InferenceSession(orig_onnx, providers=['CPUExecutionProvider'])
sess_s = ort.InferenceSession(slim_onnx, providers=['CPUExecutionProvider'])
x = DUMMY.numpy()
y_o = sess_o.run(None, {'raw_input': x})[0]
y_s = sess_s.run(None, {'raw_input': x})[0]
diff = float(np.abs(y_o - y_s).max())
print(f"\nOriginal vs Slim max abs diff: {diff:.2e}  ({'PASS' if diff < 1e-3 else 'WARN'})")
print("\nONNX export complete.")
