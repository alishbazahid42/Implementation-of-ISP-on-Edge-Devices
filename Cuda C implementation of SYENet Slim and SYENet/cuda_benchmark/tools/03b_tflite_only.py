"""
Run TFLite benchmarks and generate all reports/CSV.
Loads cached PyTorch results from the previous run and adds TFLite rows.
"""
import sys, os, time, csv, statistics, math, warnings
warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
from PIL import Image
import cv2

sys.path.insert(0, r'C:\Users\aujla\Desktop\archive\syenet')

ARCHIVE     = r'C:\Users\aujla\Desktop\archive'
BENCH_DIR   = os.path.join(ARCHIVE, 'cuda_benchmark')
RESULTS_DIR = os.path.join(BENCH_DIR, 'results')
INP_DIR     = os.path.join(ARCHIVE, 'dataset', 'test', 'mediatek_raw')
GT_DIR      = os.path.join(ARCHIVE, 'dataset', 'test', 'fujifilm')
os.makedirs(RESULTS_DIR, exist_ok=True)

WARMUP = 20

def bayer2rggb(raw):
    h, w = raw.shape
    img  = raw.reshape(h//2, 2, w//2, 2).transpose([1, 3, 0, 2]).reshape([-1, h//2, w//2])
    return img.astype(np.float32) / 4095.

def load_raw(path):
    raw = np.array(Image.open(path))
    return bayer2rggb(raw)

def load_gt(path):
    return np.array(Image.open(path).convert('RGB')).astype(np.float32) / 255.

def compute_psnr(a, b):
    mse = float(((a.astype(np.float64) - b.astype(np.float64))**2).mean())
    if mse == 0: return 100.0
    return 10. * math.log10(1. / mse)

def compute_ssim(img1, img2):
    C1, C2 = 0.01**2, 0.03**2
    vals   = []
    for c in range(3):
        i1 = img1[:,:,c].astype(np.float64)
        i2 = img2[:,:,c].astype(np.float64)
        mu1 = cv2.GaussianBlur(i1, (11,11), 1.5)
        mu2 = cv2.GaussianBlur(i2, (11,11), 1.5)
        s1  = cv2.GaussianBlur(i1**2, (11,11), 1.5) - mu1**2
        s2  = cv2.GaussianBlur(i2**2, (11,11), 1.5) - mu2**2
        s12 = cv2.GaussianBlur(i1*i2, (11,11), 1.5) - mu1*mu2
        vals.append(float(((2*mu1*mu2+C1)*(2*s12+C2)/((mu1**2+mu2**2+C1)*(s1+s2+C2))).mean()))
    return float(np.mean(vals))

def compute_metrics(pred, gt):
    diff = pred.astype(np.float64) - gt.astype(np.float64)
    mse  = float((diff**2).mean())
    mae  = float(np.abs(diff).mean())
    return compute_psnr(pred,gt), compute_ssim(pred,gt), mse, mae, math.sqrt(mse)

fnames   = sorted(f for f in os.listdir(INP_DIR) if f.endswith('.png'))
gt_names = sorted(f for f in os.listdir(GT_DIR)  if f.endswith('.png'))
inp_map  = {os.path.splitext(f)[0]: f for f in fnames}
gt_map   = {os.path.splitext(f)[0]: f for f in gt_names}
stems    = sorted(set(inp_map) & set(gt_map))
N        = len(stems)

def run_tflite_benchmark(tflite_path, model_label):
    import tensorflow as tf
    interp = tf.lite.Interpreter(model_path=tflite_path)
    interp.allocate_tensors()
    in_d  = interp.get_input_details()[0]
    out_d = interp.get_output_details()[0]
    allocated_shape = list(in_d['shape'])
    rows = []
    print(f"\n[{model_label}] Running {N} images (warmup={WARMUP}) ...")
    for i, stem in enumerate(stems):
        raw = load_raw(os.path.join(INP_DIR, inp_map[stem]))
        gt  = load_gt(os.path.join(GT_DIR,  gt_map[stem]))
        inp_nhwc = raw.transpose(1, 2, 0)[np.newaxis].astype(np.float32)
        if list(inp_nhwc.shape) != allocated_shape:
            interp.resize_tensor_input(in_d['index'], inp_nhwc.shape)
            interp.allocate_tensors()
            in_d  = interp.get_input_details()[0]
            out_d = interp.get_output_details()[0]
            allocated_shape = list(inp_nhwc.shape)
        t0 = time.perf_counter()
        interp.set_tensor(in_d['index'], inp_nhwc)
        interp.invoke()
        out = interp.get_tensor(out_d['index'])
        t1 = time.perf_counter()
        total = (t1 - t0) * 1000.
        out_hwc = np.squeeze(out).clip(0, 1).astype(np.float32)
        psnr, ssim, mse, mae, rmse = compute_metrics(out_hwc, gt)
        if i >= WARMUP:
            rows.append({
                'image_name': stem,
                'psnr': psnr, 'ssim': ssim, 'mse': mse, 'mae': mae, 'rmse': rmse,
                'h2d_time_ms': 0., 'preprocess_time_ms': 0.,
                'inference_time_ms': total, 'postprocess_time_ms': 0.,
                'd2h_time_ms': 0., 'total_latency_ms': total,
            })
        if (i+1) % 200 == 0:
            print(f"  {i+1}/{N}  PSNR={psnr:.2f}  time={total:.2f}ms")
    return rows

rows_orig_tfl = run_tflite_benchmark(os.path.join(ARCHIVE,'original_model.tflite'), 'Original TFLite CPU')
rows_slim_tfl = run_tflite_benchmark(os.path.join(ARCHIVE,'slim_model.tflite'),     'Slim TFLite CPU')

# Append to per_image_results.csv
csv_path = os.path.join(RESULTS_DIR, 'per_image_results.csv')
existing = os.path.exists(csv_path)

# Remove old TFLite rows if re-running
rows_keep = []
if existing:
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if 'tflite' not in row.get('model',''):
                rows_keep.append(row)

csv_cols = ['model','image_name','psnr','ssim','mse','mae','rmse',
            'h2d_time_ms','preprocess_time_ms','inference_time_ms',
            'postprocess_time_ms','d2h_time_ms','total_latency_ms']

with open(csv_path, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=csv_cols, extrasaction='ignore')
    w.writeheader()
    for row in rows_keep:
        w.writerow(row)
    for model_label, rows in [('original_tflite', rows_orig_tfl), ('slim_tflite', rows_slim_tfl)]:
        for r in rows:
            w.writerow({'model': model_label, **r})
print(f"Updated: {csv_path}")

# Compute summaries and append to summary_metrics.csv
def summarize_tfl(rows, label, size_kb, params):
    psnrs = [r['psnr'] for r in rows]; ssims = [r['ssim'] for r in rows]
    mses  = [r['mse']  for r in rows]; maes  = [r['mae']  for r in rows]
    rmses = [r['rmse'] for r in rows]; times = [r['total_latency_ms'] for r in rows]
    return {
        'label': label, 'size_kb': size_kb, 'params': params,
        'avg_psnr': statistics.mean(psnrs), 'avg_ssim': statistics.mean(ssims),
        'avg_mse':  statistics.mean(mses),  'avg_mae':  statistics.mean(maes),
        'avg_rmse': statistics.mean(rmses),
        'lat_avg': statistics.mean(times), 'lat_min': min(times), 'lat_max': max(times),
        'lat_median': statistics.median(times),
        'lat_std': statistics.stdev(times) if len(times)>1 else 0.,
        'fps': 1000./statistics.mean(times),
        'total_images': len(rows), 'total_time_s': sum(times)/1000.,
        'avg_sm_util': float('nan'), 'peak_sm_util': float('nan'),
        'avg_mem_mb':  float('nan'), 'peak_mem_mb':  float('nan'),
        'avg_power_w': float('nan'), 'avg_temp_c':   float('nan'),
    }

orig_tfl_kb = os.path.getsize(os.path.join(ARCHIVE,'original_model.tflite'))/1024
slim_tfl_kb = os.path.getsize(os.path.join(ARCHIVE,'slim_model.tflite'))/1024
S_ORIG_TFL  = summarize_tfl(rows_orig_tfl, 'Original TFLite CPU', orig_tfl_kb, 110984)
S_SLIM_TFL  = summarize_tfl(rows_slim_tfl, 'Slim TFLite CPU',     slim_tfl_kb,  5640)

sum_path = os.path.join(RESULTS_DIR, 'summary_metrics.csv')
sum_rows = []
if os.path.exists(sum_path):
    with open(sum_path) as f:
        for row in csv.DictReader(f):
            if 'TFLite' not in row.get('label',''):
                sum_rows.append(row)

sum_fields = ['label','size_kb','params','avg_psnr','avg_ssim','avg_mse','avg_mae','avg_rmse',
              'lat_avg','lat_min','lat_max','lat_median','lat_std','fps',
              'total_images','total_time_s','avg_sm_util','peak_sm_util',
              'avg_mem_mb','peak_mem_mb','avg_power_w','avg_temp_c']

def fmt(v):
    if isinstance(v, float) and math.isnan(v): return 'N/A'
    if isinstance(v, float): return f'{v:.6f}'
    return str(v)

with open(sum_path, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=sum_fields, extrasaction='ignore')
    w.writeheader()
    for row in sum_rows:
        w.writerow(row)
    for s in [S_ORIG_TFL, S_SLIM_TFL]:
        w.writerow({k: fmt(s.get(k, '')) for k in sum_fields})
print(f"Updated: {sum_path}")

print(f"\n  Original TFLite CPU:  PSNR={S_ORIG_TFL['avg_psnr']:.4f}  SSIM={S_ORIG_TFL['avg_ssim']:.4f}  {S_ORIG_TFL['lat_avg']:.2f} ms  {S_ORIG_TFL['fps']:.2f} FPS")
print(f"  Slim TFLite CPU   :  PSNR={S_SLIM_TFL['avg_psnr']:.4f}  SSIM={S_SLIM_TFL['avg_ssim']:.4f}  {S_SLIM_TFL['lat_avg']:.2f} ms  {S_SLIM_TFL['fps']:.2f} FPS")
