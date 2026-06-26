"""
Generate CPU vs GPU CUDA C comparison image with stats.
"""
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os

DIR = os.path.dirname(os.path.abspath(__file__))

def load(name, shape):
    path = os.path.join(DIR, name)
    arr = np.fromfile(path, dtype=np.float32).reshape(shape)
    return arr

cpu = load('output_cpu.bin', (3, 256, 256)).transpose(1, 2, 0)
gpu = load('output.bin',     (3, 256, 256)).transpose(1, 2, 0)
gt  = load('gt.bin',         (256, 256, 3))

cpu_img = (cpu.clip(0,1) * 255).astype(np.uint8)
gpu_img = (gpu.clip(0,1) * 255).astype(np.uint8)
gt_img  = (gt.clip(0,1)  * 255).astype(np.uint8)

# --- layout ---
IMG_W, IMG_H = 256, 256
PAD        = 20
HEADER     = 70   # top title bar
LABEL      = 36   # label under each image
STATS_H    = 110  # bottom stats bar

COLS = 3
W = PAD + COLS * IMG_W + (COLS - 1) * PAD + PAD
H = HEADER + IMG_H + LABEL + PAD + STATS_H

BG        = (18,  18,  28)
NAVY      = (22,  55,  100)
TEAL      = (13, 115, 119)
RED_DARK  = (160, 30,  30)
GREEN_DRK = (20, 110,  50)
WHITE     = (255, 255, 255)
LGRAY     = (180, 190, 200)
GOLD      = (230, 170,  30)

canvas = Image.new('RGB', (W, H), BG)
draw   = ImageDraw.Draw(canvas)

# ── try to load a font, fall back gracefully ──────────────────────────
def font(size):
    for name in [
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/consola.ttf",
    ]:
        try:
            return ImageFont.truetype(name, size)
        except:
            pass
    return ImageFont.load_default()

F_TITLE  = font(22)
F_LABEL  = font(15)
F_STAT   = font(14)
F_SMALL  = font(12)

# ── header bar ────────────────────────────────────────────────────────
draw.rectangle([0, 0, W, HEADER], fill=NAVY)
draw.rectangle([0, HEADER-3, W, HEADER], fill=TEAL)
title = "SYEISPNetS — CPU vs GPU CUDA C Inference"
bb = draw.textbbox((0,0), title, font=F_TITLE)
tw = bb[2] - bb[0]
draw.text(((W - tw)//2, 14), title, fill=WHITE, font=F_TITLE)
sub = "Plain C (/O2 /AVX2)  vs  CUDA C (sm_75)  |  Quadro T1000  |  Windows 11"
bb2 = draw.textbbox((0,0), sub, font=F_SMALL)
draw.text(((W - (bb2[2]-bb2[0]))//2, 44), sub, fill=(140, 170, 210), font=F_SMALL)

# ── place images + labels ─────────────────────────────────────────────
panels = [
    (cpu_img, "CPU Plain C",   "646 ms  |  1.55 FPS",  RED_DARK,   (220,60,60)),
    (gpu_img, "GPU CUDA C",    "1.56 ms  |  641 FPS",  GREEN_DRK,  (50,200,100)),
    (gt_img,  "Fujifilm GT",   "Reference output",      (50,50,80), LGRAY),
]

y_img   = HEADER + PAD // 2
y_label = y_img + IMG_H + 6

for i, (img_arr, title_lbl, sub_lbl, bg_col, txt_col) in enumerate(panels):
    x = PAD + i * (IMG_W + PAD)

    # border / background for label area
    draw.rectangle([x-2, y_img-2, x+IMG_W+1, y_label+LABEL+2], fill=bg_col, outline=txt_col, width=1)

    # paste image
    pil_img = Image.fromarray(img_arr)
    canvas.paste(pil_img, (x, y_img))

    # label
    bb = draw.textbbox((0,0), title_lbl, font=F_LABEL)
    tw = bb[2]-bb[0]
    draw.text((x + (IMG_W-tw)//2, y_label+4), title_lbl, fill=WHITE, font=F_LABEL)
    bb2 = draw.textbbox((0,0), sub_lbl, font=F_SMALL)
    tw2 = bb2[2]-bb2[0]
    draw.text((x + (IMG_W-tw2)//2, y_label+21), sub_lbl, fill=txt_col, font=F_SMALL)

# ── stats bar ─────────────────────────────────────────────────────────
y_stats = y_label + LABEL + PAD
draw.rectangle([0, y_stats, W, H], fill=(12, 20, 36))
draw.rectangle([0, y_stats, W, y_stats+2], fill=TEAL)

stats = [
    ("Speedup",      "415×",        GOLD,            "GPU is 415× faster than CPU"),
    ("CPU latency",  "646 ms",      (220, 80, 80),   "Plain C, single-threaded"),
    ("GPU latency",  "1.56 ms",     (50, 200, 100),  "CUDA kernels, parallel"),
    ("Verification", "135.8 dB",    (100, 180, 255), "PSNR CPU vs GPU — bit-identical"),
]

col_w = W // len(stats)
for i, (lbl, val, vcol, note) in enumerate(stats):
    cx = i * col_w + col_w // 2
    # value
    bb = draw.textbbox((0,0), val, font=F_TITLE)
    tw = bb[2]-bb[0]
    draw.text((cx - tw//2, y_stats + 14), val, fill=vcol, font=F_TITLE)
    # label
    bb2 = draw.textbbox((0,0), lbl, font=F_STAT)
    tw2 = bb2[2]-bb2[0]
    draw.text((cx - tw2//2, y_stats + 46), lbl, fill=WHITE, font=F_STAT)
    # note
    bb3 = draw.textbbox((0,0), note, font=F_SMALL)
    tw3 = bb3[2]-bb3[0]
    draw.text((cx - tw3//2, y_stats + 68), note, fill=LGRAY, font=F_SMALL)
    # divider
    if i > 0:
        draw.line([i*col_w, y_stats+10, i*col_w, H-10], fill=(40,60,90), width=1)

out = os.path.join(DIR, '..', 'CPU_vs_GPU_CUDA_C_comparison.png')
canvas.save(out)
print(f"Saved: {os.path.abspath(out)}")
