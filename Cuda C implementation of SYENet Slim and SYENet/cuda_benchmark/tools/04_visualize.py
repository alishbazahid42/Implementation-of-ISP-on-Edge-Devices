"""
Step 4: Generate all publication-quality plots from benchmark results.

Generates:
  psnr_distribution.png
  ssim_distribution.png
  latency_distribution.png
  gpu_memory_usage.png
  gpu_utilization.png
  model_size_comparison.png
  fps_comparison.png
  quality_vs_latency.png
  throughput_comparison.png

Run after 03_run_benchmark.py:
  py -3.10 04_visualize.py
"""
import os, csv, math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

RESULTS_DIR = r'C:\Users\aujla\Desktop\archive\cuda_benchmark\results'
PLOTS_DIR   = os.path.join(RESULTS_DIR, 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

# ── Load data ─────────────────────────────────────────────────────────────────
per_image = {}
with open(os.path.join(RESULTS_DIR, 'per_image_results.csv')) as f:
    for row in csv.DictReader(f):
        m = row['model']
        if m not in per_image: per_image[m] = []
        per_image[m].append({k: (float(v) if v and v != '' else float('nan')) if k != 'image_name' and k != 'model' else v
                              for k, v in row.items()})

summary = {}
with open(os.path.join(RESULTS_DIR, 'summary_metrics.csv')) as f:
    for row in csv.DictReader(f):
        def to_float(v):
            try: return float(v)
            except: return float('nan')
        summary[row['label']] = {k: to_float(v) if k != 'label' else v
                                 for k, v in row.items()}

MODELS = {
    'original_pytorch': 'Original\nPyTorch GPU',
    'slim_pytorch':     'Slim\nPyTorch GPU',
    'original_tflite':  'Original\nTFLite CPU',
    'slim_tflite':      'Slim\nTFLite CPU',
}
COLORS = {
    'original_pytorch': '#2563EB',
    'slim_pytorch':     '#16A34A',
    'original_tflite':  '#DC2626',
    'slim_tflite':      '#D97706',
}
STYLE = {
    'axes.spines.top': False, 'axes.spines.right': False,
    'font.family': 'DejaVu Sans', 'font.size': 11,
    'axes.labelsize': 12, 'axes.titlesize': 13,
    'figure.dpi': 150,
}
plt.rcParams.update(STYLE)

def save(fig, name):
    path = os.path.join(PLOTS_DIR, name)
    fig.savefig(path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  Saved: {name}")

# ── 1. PSNR distribution ──────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
for key, label in MODELS.items():
    if key not in per_image: continue
    vals = [r['psnr'] for r in per_image[key]]
    ax.hist(vals, bins=40, alpha=0.6, color=COLORS[key], label=label.replace('\n', ' '), edgecolor='none')
ax.set_xlabel('PSNR (dB)')
ax.set_ylabel('Number of images')
ax.set_title('PSNR distribution across 2,417 test images')
ax.legend(frameon=False)
ax.axvline(20, color='#6B7280', linestyle='--', lw=1, alpha=0.5)
save(fig, 'psnr_distribution.png')

# ── 2. SSIM distribution ──────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
for key, label in MODELS.items():
    if key not in per_image: continue
    vals = [r['ssim'] for r in per_image[key]]
    ax.hist(vals, bins=40, alpha=0.6, color=COLORS[key], label=label.replace('\n', ' '), edgecolor='none')
ax.set_xlabel('SSIM')
ax.set_ylabel('Number of images')
ax.set_title('SSIM distribution across 2,417 test images')
ax.legend(frameon=False)
save(fig, 'ssim_distribution.png')

# ── 3. Latency distribution ───────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, (group_keys, title) in zip(axes, [
    (['original_pytorch', 'slim_pytorch'], 'GPU inference latency (PyTorch)'),
    (['original_tflite',  'slim_tflite'],  'CPU inference latency (TFLite)'),
]):
    for key in group_keys:
        if key not in per_image: continue
        vals = [r['inference_time_ms'] for r in per_image[key]]
        ax.hist(vals, bins=40, alpha=0.65, color=COLORS[key],
                label=MODELS[key].replace('\n', ' '), edgecolor='none')
    ax.set_xlabel('Inference time (ms)')
    ax.set_ylabel('Count')
    ax.set_title(title)
    ax.legend(frameon=False)
fig.tight_layout()
save(fig, 'latency_distribution.png')

# ── 4. GPU memory usage ───────────────────────────────────────────────────────
keys_gpu = ['original_pytorch', 'slim_pytorch']
mem_vals  = {k: [r.get('avg_mem_mb', float('nan')) for r in per_image.get(k, [])] for k in keys_gpu}
fig, ax   = plt.subplots(figsize=(8, 4))
for key in keys_gpu:
    vals = [v for v in mem_vals[key] if not math.isnan(v)]
    if vals:
        ax.plot(range(len(vals)), vals, color=COLORS[key],
                lw=1, alpha=0.8, label=MODELS[key].replace('\n', ' '))
ax.set_xlabel('Image index')
ax.set_ylabel('GPU memory used (MB)')
ax.set_title('GPU memory usage over inference run')
ax.legend(frameon=False)
save(fig, 'gpu_memory_usage.png')

# ── 5. GPU utilization ────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
plotted = False
for key in keys_gpu:
    vals = [r.get('sm_util', float('nan')) for r in per_image.get(key, [])]
    clean = [v for v in vals if not math.isnan(v)]
    if clean:
        ax.plot(range(len(clean)), clean, color=COLORS[key],
                lw=1, alpha=0.8, label=MODELS[key].replace('\n', ' '))
        plotted = True
if not plotted:
    ax.text(0.5, 0.5, 'GPU utilization data not available\n(install pynvml: pip install pynvml)',
            ha='center', va='center', transform=ax.transAxes, color='#6B7280')
ax.set_xlabel('Image index')
ax.set_ylabel('SM utilization (%)')
ax.set_title('GPU SM utilization during inference')
ax.legend(frameon=False)
save(fig, 'gpu_utilization.png')

# ── 6. Model size comparison ──────────────────────────────────────────────────
size_data = {
    'Original\nPyTorch': 524.9,
    'Slim\nPyTorch':     26.6,
    'Original\nTFLite':  419.1,
    'Slim\nTFLite':      29.1,
}
fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(size_data.keys(), size_data.values(),
              color=['#2563EB','#16A34A','#DC2626','#D97706'],
              width=0.5, edgecolor='none')
for bar, val in zip(bars, size_data.values()):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
            f'{val:.1f} KB', ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.set_ylabel('File size (KB)')
ax.set_title('Model size comparison')
ax.set_ylim(0, max(size_data.values()) * 1.2)
save(fig, 'model_size_comparison.png')

# ── 7. FPS comparison ────────────────────────────────────────────────────────
sum_labels = {
    'Original PyTorch GPU': 'original_pytorch',
    'Slim PyTorch GPU':     'slim_pytorch',
    'Original TFLite CPU':  'original_tflite',
    'Slim TFLite CPU':      'slim_tflite',
}
fps_data = {}
for slabel, key in sum_labels.items():
    s = summary.get(slabel)
    if s: fps_data[MODELS[key].replace('\n', ' ')] = s['fps']

fig, ax = plt.subplots(figsize=(9, 5))
model_names = list(fps_data.keys())
fps_vals    = list(fps_data.values())
bar_colors  = [COLORS['original_pytorch'], COLORS['slim_pytorch'],
               COLORS['original_tflite'],  COLORS['slim_tflite']]
bars = ax.bar(model_names, fps_vals, color=bar_colors[:len(fps_vals)], width=0.5)
for bar, val in zip(bars, fps_vals):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'{val:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
ax.set_ylabel('FPS (frames per second)')
ax.set_title('Throughput comparison (FPS)')
ax.set_ylim(0, max(fps_vals) * 1.2)
save(fig, 'fps_comparison.png')

# ── 8. Quality vs Latency scatter ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 6))
for key, slabel in zip(['original_pytorch','slim_pytorch','original_tflite','slim_tflite'],
                        ['Original PyTorch GPU','Slim PyTorch GPU','Original TFLite CPU','Slim TFLite CPU']):
    s = summary.get(slabel)
    if s:
        ax.scatter(s['lat_avg'], s['avg_psnr'], s=200, color=COLORS[key], zorder=5,
                   label=MODELS[key].replace('\n', ' '), edgecolors='white', linewidths=1.5)
        ax.annotate(MODELS[key].replace('\n', ' '),
                    (s['lat_avg'], s['avg_psnr']),
                    textcoords='offset points', xytext=(8, 4), fontsize=9,
                    color=COLORS[key])
ax.set_xlabel('Average latency (ms)')
ax.set_ylabel('Average PSNR (dB)')
ax.set_title('Image quality vs inference latency\n(lower-left = faster & better quality)')
ax.legend(frameon=False, fontsize=9)
save(fig, 'quality_vs_latency.png')

# ── 9. Throughput comparison (grouped bar) ───────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(2)
w = 0.3
orig_fps = [
    summary.get('Original PyTorch GPU', {}).get('fps', 0),
    summary.get('Original TFLite CPU',  {}).get('fps', 0),
]
slim_fps = [
    summary.get('Slim PyTorch GPU',  {}).get('fps', 0),
    summary.get('Slim TFLite CPU',   {}).get('fps', 0),
]
b1 = ax.bar(x - w/2, orig_fps, w, label='Original', color='#2563EB', alpha=0.85)
b2 = ax.bar(x + w/2, slim_fps, w, label='Slim',     color='#16A34A', alpha=0.85)
for bar in [*b1, *b2]:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 0.5,
            f'{h:.1f}', ha='center', va='bottom', fontsize=10)
ax.set_xticks(x)
ax.set_xticklabels(['GPU (PyTorch)', 'CPU (TFLite)'])
ax.set_ylabel('FPS')
ax.set_title('Throughput: Original vs Slim')
ax.legend(frameon=False)
save(fig, 'throughput_comparison.png')

print(f"\nAll plots saved to: {PLOTS_DIR}")
