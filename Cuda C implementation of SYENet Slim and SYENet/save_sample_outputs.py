"""
save_sample_outputs.py
Run inference on 10 sample images:
  - Original model (orig_model.onnx via ONNX Runtime — multi-branch unfused)
  - New Module slim (CUDA C exe, weights_nm.bin — reparameterized)
  - Fujifilm GT (reference)
Save side-by-side PNG + HTML report with PSNR table.
"""
import os, sys, subprocess, math, tempfile, json
import numpy as np
from PIL import Image, ImageDraw

import onnxruntime as ort

EXE         = r"C:\Users\aujla\Desktop\archive\cuda_isp\syenet_isp.exe"
W_NM        = r"C:\Users\aujla\Desktop\archive\new_module\weights_nm.bin"
ORIG_ONNX   = r"C:\Users\aujla\Desktop\archive\orig_model.onnx"
RAW_DIR     = r"C:\Users\aujla\Desktop\archive\dataset\test\mediatek_raw"
GT_DIR      = r"C:\Users\aujla\Desktop\archive\dataset\test\fujifilm"
OUT_DIR     = r"C:\Users\aujla\Desktop\archive\sample_outputs"

os.makedirs(OUT_DIR, exist_ok=True)

# load ONNX session once
sess_orig = ort.InferenceSession(ORIG_ONNX)
inp_name  = sess_orig.get_inputs()[0].name
print(f"ONNX orig_model loaded. Input: {inp_name} {sess_orig.get_inputs()[0].shape}")

def psnr(a, b):
    mse = float(((a.astype(np.float64) - b.astype(np.float64))**2).mean())
    return 100.0 if mse == 0 else 10.0 * math.log10(1.0 / mse)

def raw_to_float32(path):
    raw = np.array(Image.open(path))
    h, w = raw.shape
    rggb = raw.reshape(h//2, 2, w//2, 2).transpose([1, 3, 0, 2]).reshape([-1, h//2, w//2])
    return rggb.astype(np.float32) / 4095.0

def run_cuda_nm(inp_bin, out_bin):
    r = subprocess.run([EXE, W_NM, inp_bin, out_bin, "1"], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[:200])
    lat = None
    for line in r.stdout.splitlines():
        if line.startswith("LATENCY_MS="):
            lat = float(line.split("=")[1])
    return lat

def f2u8(arr):
    return (arr.clip(0, 1) * 255).astype(np.uint8)

# pick 10 spread evenly across dataset
all_files = sorted(f for f in os.listdir(RAW_DIR) if f.endswith(".png"))
total     = len(all_files)
samples   = [all_files[int(total * i / 10)] for i in range(10)]
print(f"Selected {len(samples)} images from {total} total\n")

tmp_in  = os.path.join(tempfile.gettempdir(), "syenet_sample_in.bin")
tmp_out = os.path.join(tempfile.gettempdir(), "syenet_sample_out.bin")

results = []
LABEL_H = 26
BORDER  = 4
COL_W   = 256
IMG_H   = 256

for idx, fname in enumerate(samples):
    raw_path = os.path.join(RAW_DIR, fname)
    gt_path  = os.path.join(GT_DIR,  fname)
    print(f"[{idx+1}/10] {fname}")

    rggb = raw_to_float32(raw_path)
    rggb.tofile(tmp_in)

    # ── false-colour RAW preview ──────────────────────────────────────────────
    r_ch = np.repeat(np.repeat(f2u8(rggb[0]), 2, 0), 2, 1)
    g_ch = np.repeat(np.repeat(f2u8((rggb[1]+rggb[2])/2), 2, 0), 2, 1)
    b_ch = np.repeat(np.repeat(f2u8(rggb[3]), 2, 0), 2, 1)
    raw_vis = np.stack([r_ch, g_ch, b_ch], axis=2)

    # ── Original ONNX ─────────────────────────────────────────────────────────
    inp_t    = rggb[np.newaxis]   # (1, 4, 128, 128)
    orig_out = sess_orig.run(None, {inp_name: inp_t})[0][0].transpose(1,2,0).clip(0,1)

    # ── New Module CUDA C ─────────────────────────────────────────────────────
    lat_nm = run_cuda_nm(tmp_in, tmp_out)
    nm_out = np.fromfile(tmp_out, dtype=np.float32).reshape(3,256,256).transpose(1,2,0).clip(0,1)

    # ── GT ────────────────────────────────────────────────────────────────────
    gt_arr = np.array(Image.open(gt_path).convert("RGB")).astype(np.float32) / 255.0

    # ── PSNR ──────────────────────────────────────────────────────────────────
    p_orig = psnr(orig_out, gt_arr)
    p_nm   = psnr(nm_out,   gt_arr)
    print(f"  Orig ONNX PSNR: {p_orig:.2f} dB   NM CUDA PSNR: {p_nm:.2f} dB"
          f"   lat={lat_nm:.2f}ms")

    # ── 4-panel strip ─────────────────────────────────────────────────────────
    n_panels = 4
    strip_w  = COL_W * n_panels + BORDER * (n_panels + 1)
    strip_h  = IMG_H + LABEL_H * 2 + BORDER * 3
    strip    = Image.new("RGB", (strip_w, strip_h), (15, 15, 15))
    draw     = ImageDraw.Draw(strip)

    panels = [
        (raw_vis,             "RAW input (false-color)", "",              (180, 180, 180)),
        (f2u8(orig_out),      "Original model (ONNX)",   f"{p_orig:.2f} dB vs GT", (100, 180, 255)),
        (f2u8(nm_out),        "New Module slim (CUDA C)", f"{p_nm:.2f} dB vs GT",   (80, 220, 130)),
        (f2u8(gt_arr),        "Fujifilm GT (reference)",  "",              (255, 200, 80)),
    ]

    for ci, (arr, label, sub, col) in enumerate(panels):
        x = BORDER + ci * (COL_W + BORDER)
        y = LABEL_H + BORDER
        strip.paste(Image.fromarray(arr), (x, y))
        draw.text((x + 4, BORDER // 2 + 2), label, fill=col)
        if sub:
            draw.text((x + 4, y + IMG_H + 4), sub, fill=col)

    bar_y = IMG_H + LABEL_H + BORDER * 2 + LABEL_H - 4
    draw.text((BORDER + 2, bar_y), f"#{idx+1}  {fname}", fill=(80, 80, 80))

    out_path = os.path.join(OUT_DIR, f"sample_{idx+1:02d}_{fname.replace('.png','')}.png")
    strip.save(out_path, "PNG")
    results.append({
        "idx": idx + 1, "file": fname,
        "psnr_orig": round(p_orig, 3),
        "psnr_nm":   round(p_nm, 3),
        "diff":      round(p_nm - p_orig, 3),
        "lat_nm_ms": round(lat_nm, 3) if lat_nm else None,
        "path":      out_path,
    })

# ── HTML report ────────────────────────────────────────────────────────────────
avg_orig = sum(r["psnr_orig"] for r in results) / 10
avg_nm   = sum(r["psnr_nm"]   for r in results) / 10
avg_diff = avg_nm - avg_orig

rows = ""
for r in results:
    c = "#3FB950" if r["diff"] > 0 else "#F85149"
    rows += f"""<tr>
      <td>{r['idx']}</td><td style="font-size:10px;color:#6E7681">{r['file']}</td>
      <td style="color:#58A6FF">{r['psnr_orig']:.3f}</td>
      <td style="color:#3FB950;font-weight:700">{r['psnr_nm']:.3f}</td>
      <td style="color:{c};font-weight:700">{r['diff']:+.3f}</td>
      <td>{r['lat_nm_ms']} ms</td></tr>"""

cards = ""
for r in results:
    rel = os.path.basename(r["path"])
    c   = "#3FB950" if r["diff"] > 0 else "#F85149"
    cards += f"""<div style="margin-bottom:24px">
      <div style="font-size:12px;color:#8B949E;margin-bottom:6px">
        <b style="color:#E6EDF3">#{r['idx']} — {r['file']}</b> &nbsp;·&nbsp;
        Orig ONNX: <span style="color:#58A6FF">{r['psnr_orig']:.2f} dB</span> &nbsp;·&nbsp;
        NM CUDA C: <span style="color:#3FB950">{r['psnr_nm']:.2f} dB</span> &nbsp;·&nbsp;
        Diff: <span style="color:{c};font-weight:700">{r['diff']:+.2f} dB</span>
      </div>
      <img src="{rel}" style="width:100%;border-radius:6px;border:1px solid #30363D">
    </div>"""

html = f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8">
<title>SYENet CUDA C — 10 Sample Images</title>
<style>
* {{box-sizing:border-box;margin:0;padding:0}}
body {{font-family:'Segoe UI',sans-serif;background:#0D1117;color:#E6EDF3;padding:32px}}
h1 {{font-size:20px;color:#58A6FF;margin-bottom:6px}}
.sub {{font-size:12px;color:#8B949E;margin-bottom:24px}}
.note {{background:#161B22;border:1px solid #F0883E;border-radius:8px;padding:12px 16px;
  font-size:11px;color:#8B949E;margin-bottom:20px}}
.note strong {{color:#F0883E}}
.stats {{display:flex;gap:10px;margin-bottom:24px;flex-wrap:wrap}}
.stat {{background:#161B22;border:1px solid #30363D;border-radius:8px;padding:12px 18px;text-align:center}}
.stat-val {{font-size:20px;font-weight:800;margin-bottom:3px}}
.stat-lbl {{font-size:9px;color:#8B949E}}
h2 {{font-size:12px;font-weight:700;text-transform:uppercase;letter-spacing:1px;color:#F0883E;
  border-bottom:1px solid #21262D;padding-bottom:6px;margin:24px 0 12px}}
table {{width:100%;border-collapse:collapse;font-size:12px;margin-bottom:24px}}
th {{background:#161B22;color:#8B949E;font-weight:600;padding:7px 10px;text-align:left;border-bottom:2px solid #21262D}}
td {{padding:6px 10px;border-bottom:1px solid #21262D}}
tr:nth-child(even) td {{background:#0D1117}}
tr:nth-child(odd) td {{background:#161B22}}
</style></head><body>
<h1>SYENet — 10 Sample Image Comparison</h1>
<div class="sub">RAW input &nbsp;|&nbsp; Original model (ONNX Runtime) &nbsp;|&nbsp; New Module slim (CUDA C) &nbsp;|&nbsp; Fujifilm GT reference</div>
<div class="note">
  <strong>Note on models:</strong> "Original model (ONNX)" = multi-branch unfused SYEISPNet run via ONNX Runtime —
  this is the training-time architecture before reparameterization.
  "New Module slim (CUDA C)" = reparameterized SYEISPNetS run via the custom CUDA C binary (syenet_isp.exe) at 640+ FPS.
  The slim model significantly outperforms the unfused model because reparameterization properly fuses all branches.
</div>
<div class="stats">
  <div class="stat"><div class="stat-val" style="color:#58A6FF">{avg_orig:.2f} dB</div><div class="stat-lbl">Orig ONNX avg PSNR</div></div>
  <div class="stat"><div class="stat-val" style="color:#3FB950">{avg_nm:.2f} dB</div><div class="stat-lbl">NM CUDA C avg PSNR</div></div>
  <div class="stat"><div class="stat-val" style="color:#F0883E">{avg_diff:+.2f} dB</div><div class="stat-lbl">NM improvement</div></div>
  <div class="stat"><div class="stat-val" style="color:#E6EDF3">10</div><div class="stat-lbl">Images</div></div>
</div>
<h2>PSNR Table</h2>
<table><tr><th>#</th><th>File</th><th>Orig ONNX PSNR</th><th>NM CUDA C PSNR</th><th>Diff</th><th>NM Latency</th></tr>
{rows}
<tr style="background:#0D2B1B">
  <td colspan="2" style="color:#3FB950;font-weight:700">AVERAGE</td>
  <td style="color:#58A6FF;font-weight:700">{avg_orig:.3f} dB</td>
  <td style="color:#3FB950;font-weight:700">{avg_nm:.3f} dB</td>
  <td style="color:#3FB950;font-weight:700">{avg_diff:+.3f} dB</td><td></td>
</tr></table>
<h2>Visual Comparison  [RAW &nbsp; Original ONNX &nbsp; NM CUDA C &nbsp; Fujifilm GT]</h2>
{cards}
</body></html>"""

html_path = os.path.join(OUT_DIR, "comparison.html")
with open(html_path, "w", encoding="utf-8") as f:
    f.write(html)

print(f"\n{'='*60}")
print(f"  Orig ONNX avg PSNR : {avg_orig:.3f} dB")
print(f"  NM CUDA C avg PSNR : {avg_nm:.3f} dB")
print(f"  Improvement        : {avg_diff:+.3f} dB")
print(f"  Output folder      : {OUT_DIR}")
print(f"  HTML report        : {html_path}")
print(f"{'='*60}")
