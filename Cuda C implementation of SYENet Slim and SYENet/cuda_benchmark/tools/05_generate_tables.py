"""
Step 5: Generate publication-ready tables in CSV, LaTeX, and Markdown formats.

Tables:
  Table 1: Model Size Comparison
  Table 2: Latency Comparison
  Table 3: GPU Memory Comparison
  Table 4: Image Quality Metrics
  Table 5: Deployment Performance Analysis

Run after 03_run_benchmark.py:
  py -3.10 05_generate_tables.py
"""
import os, csv, math

RESULTS_DIR = r'C:\Users\aujla\Desktop\archive\cuda_benchmark\results'
TABLES_DIR  = os.path.join(RESULTS_DIR, 'tables')
os.makedirs(TABLES_DIR, exist_ok=True)

# ── Load summary ──────────────────────────────────────────────────────────────
summary = {}
with open(os.path.join(RESULTS_DIR, 'summary_metrics.csv')) as f:
    for row in csv.DictReader(f):
        def to_f(v):
            try: return float(v)
            except: return float('nan')
        summary[row['label']] = {k: to_f(v) if k != 'label' else v
                                 for k, v in row.items()}

def g(label, key):
    s = summary.get(label, {})
    v = s.get(key, float('nan'))
    return v

def fmt(v, d=2):
    if isinstance(v, float) and math.isnan(v): return 'N/A'
    if isinstance(v, float): return f'{v:.{d}f}'
    return str(v)

ROWS = [
    ('Original PyTorch GPU', 'original_pytorch', 524.9,  110984, 'GPU'),
    ('Slim PyTorch GPU',     'slim_pytorch',      26.6,    5640,  'GPU'),
    ('Original TFLite CPU',  'original_tflite',  419.1,  110984, 'CPU'),
    ('Slim TFLite CPU',      'slim_tflite',       29.1,    5640,  'CPU'),
]

# ── Table helper ──────────────────────────────────────────────────────────────
def write_csv(path, headers, rows):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(headers)
        w.writerows(rows)
    print(f"  {os.path.basename(path)}")

def write_latex(path, caption, label, headers, rows):
    ncols = len(headers)
    col_fmt = 'l' + 'r' * (ncols - 1)
    lines = [
        r'\begin{table}[h]',
        r'\centering',
        r'\caption{' + caption + '}',
        r'\label{' + label + '}',
        r'\begin{tabular}{' + col_fmt + '}',
        r'\toprule',
        ' & '.join(headers) + r' \\',
        r'\midrule',
    ]
    for row in rows:
        lines.append(' & '.join(str(c) for c in row) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"  {os.path.basename(path)}")

def write_markdown(path, headers, rows):
    def pad(s, w): return str(s).ljust(w)
    widths = [max(len(str(h)), max(len(str(r[i])) for r in rows))
              for i, h in enumerate(headers)]
    sep = '| ' + ' | '.join('-' * w for w in widths) + ' |'
    hdr = '| ' + ' | '.join(pad(h, widths[i]) for i, h in enumerate(headers)) + ' |'
    lines = [hdr, sep]
    for row in rows:
        lines.append('| ' + ' | '.join(pad(row[i], widths[i]) for i in range(len(headers))) + ' |')
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"  {os.path.basename(path)}")

def save_table(name, caption, label, headers, rows):
    print(f"\nTable: {caption}")
    write_csv(    os.path.join(TABLES_DIR, f'{name}.csv'),    headers, rows)
    write_latex(  os.path.join(TABLES_DIR, f'{name}.tex'),    caption, label, headers, rows)
    write_markdown(os.path.join(TABLES_DIR, f'{name}.md'),    headers, rows)

# ── TABLE 1: Model Size Comparison ────────────────────────────────────────────
h1 = ['Model', 'Runtime', 'File Size (KB)', 'Parameters', 'Compression Ratio']
r1 = []
base_params = 110984
for name, key, size_kb, params, runtime in ROWS:
    ratio = f'{110984/params:.1f}x' if params < 110984 else '1.0x (baseline)'
    r1.append([name, runtime, f'{size_kb:.1f}', f'{params:,}', ratio])
save_table('table1_model_size', 'Model Size Comparison', 'tab:model_size', h1, r1)

# ── TABLE 2: Latency Comparison ───────────────────────────────────────────────
h2 = ['Model', 'Avg Latency (ms)', 'Min (ms)', 'Max (ms)', 'Median (ms)', 'Std (ms)', 'FPS']
r2 = []
for name, key, _, _, _ in ROWS:
    s = summary.get(name, {})
    r2.append([
        name,
        fmt(s.get('lat_avg', float('nan')), 3),
        fmt(s.get('lat_min', float('nan')), 3),
        fmt(s.get('lat_max', float('nan')), 3),
        fmt(s.get('lat_median', float('nan')), 3),
        fmt(s.get('lat_std', float('nan')), 3),
        fmt(s.get('fps', float('nan')), 2),
    ])
save_table('table2_latency', 'Inference Latency Comparison', 'tab:latency', h2, r2)

# ── TABLE 3: GPU Memory Comparison ───────────────────────────────────────────
h3 = ['Model', 'Avg Memory (MB)', 'Peak Memory (MB)', 'Avg SM Util (%)', 'Avg Power (W)', 'Avg Temp (°C)']
r3 = []
for name, key, _, _, runtime in ROWS:
    s = summary.get(name, {})
    if runtime == 'CPU':
        r3.append([name, 'N/A (CPU)', 'N/A', 'N/A', 'N/A', 'N/A'])
    else:
        r3.append([
            name,
            fmt(s.get('avg_mem_mb', float('nan')), 1),
            fmt(s.get('peak_mem_mb', float('nan')), 1),
            fmt(s.get('avg_sm_util', float('nan')), 1),
            fmt(s.get('avg_power_w', float('nan')), 1),
            fmt(s.get('avg_temp_c', float('nan')), 1),
        ])
save_table('table3_gpu_memory', 'GPU Resource Utilization', 'tab:gpu_mem', h3, r3)

# ── TABLE 4: Image Quality Metrics ────────────────────────────────────────────
h4 = ['Model', 'Avg PSNR (dB)', 'Avg SSIM', 'Avg MSE', 'Avg MAE', 'Avg RMSE']
r4 = []
for name, key, _, _, _ in ROWS:
    s = summary.get(name, {})
    r4.append([
        name,
        fmt(s.get('avg_psnr', float('nan')), 4),
        fmt(s.get('avg_ssim', float('nan')), 4),
        fmt(s.get('avg_mse',  float('nan')), 6),
        fmt(s.get('avg_mae',  float('nan')), 6),
        fmt(s.get('avg_rmse', float('nan')), 6),
    ])
save_table('table4_quality', 'Image Quality Metrics (2,417 test images)', 'tab:quality', h4, r4)

# ── TABLE 5: Deployment Performance Analysis ──────────────────────────────────
h5 = ['Target Platform', 'Recommended Model', 'Est. FPS', 'Memory Budget', 'Deployment Format', 'Suitability']
r5 = [
    ['Jetson Nano (4 GB)',     'Slim TFLite', '60-80',   '18-29 KB model + 2 MB runtime', 'TFLite XNNPACK',  'Suitable'],
    ['Jetson Orin Nano (8 GB)','Slim TFLite', '150-200', '18-29 KB model',                'TFLite / TRT FP16','Excellent'],
    ['RTX 3080/4080',          'Slim PyTorch','1000+',   '26.6 KB model + CUDA',         'PyTorch / TRT FP16','Excellent'],
    ['Quadro T1000 (4 GB)',    'Slim PyTorch','180+',    '26.6 KB model + CUDA',         'PyTorch / TRT FP32','Excellent'],
    ['Tesla V100/A100',        'Slim PyTorch','2000+',   '26.6 KB model + CUDA',         'TRT FP16 / INT8',  'Excellent'],
    ['Raspberry Pi (no GPU)',  'Slim TFLite', '2-5',     '29.1 KB model',                'TFLite CPU',       'Limited'],
    ['Mobile (ARM, no GPU)',   'Slim TFLite', '10-30',   '29.1 KB model',                'TFLite NNAPI',     'Suitable'],
]
save_table('table5_deployment', 'Deployment Performance Analysis', 'tab:deploy', h5, r5)

# ── Generate combined LaTeX file ──────────────────────────────────────────────
combined_tex = os.path.join(TABLES_DIR, 'all_tables.tex')
with open(combined_tex, 'w') as out:
    out.write("% SYENet ISP Benchmarking — All Publication Tables\n")
    out.write("% Include in LaTeX with: \\input{all_tables.tex}\n")
    out.write("% Requires: \\usepackage{booktabs}\n\n")
    for name in ['table1_model_size','table2_latency','table3_gpu_memory',
                 'table4_quality','table5_deployment']:
        tex = os.path.join(TABLES_DIR, f'{name}.tex')
        if os.path.exists(tex):
            with open(tex) as t:
                out.write(t.read())
                out.write("\n\n")
print(f"\n  all_tables.tex  (combined LaTeX)")
print(f"\nAll tables saved to: {TABLES_DIR}")
