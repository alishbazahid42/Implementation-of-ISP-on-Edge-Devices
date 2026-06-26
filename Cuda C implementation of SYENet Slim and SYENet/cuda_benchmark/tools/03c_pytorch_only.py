"""Run PyTorch GPU benchmark only and append results to CSV."""
import sys, os, time, csv, statistics, math, warnings
warnings.filterwarnings('ignore')
import numpy as np, torch
from PIL import Image
import cv2

sys.path.insert(0, r'C:\Users\aujla\Desktop\archive\syenet')
from model.isp import SYEISPNet

ARCHIVE     = r'C:\Users\aujla\Desktop\archive'
RESULTS_DIR = os.path.join(ARCHIVE, 'cuda_benchmark', 'results')
INP_DIR     = os.path.join(ARCHIVE, 'dataset', 'test', 'mediatek_raw')
GT_DIR      = os.path.join(ARCHIVE, 'dataset', 'test', 'fujifilm')
DEVICE = torch.device('cuda')
WARMUP = 20

def bayer2rggb(raw):
    h, w = raw.shape
    return (raw.reshape(h//2,2,w//2,2).transpose([1,3,0,2]).reshape([-1,h//2,w//2])).astype(np.float32)/4095.

def compute_psnr(a,b):
    mse=float(((a.astype(np.float64)-b.astype(np.float64))**2).mean())
    return 100. if mse==0 else 10.*math.log10(1./mse)

def compute_ssim(i1,i2):
    C1,C2=0.0001,0.0009; v=[]
    for c in range(3):
        a=i1[:,:,c].astype(np.float64); b=i2[:,:,c].astype(np.float64)
        m1=cv2.GaussianBlur(a,(11,11),1.5); m2=cv2.GaussianBlur(b,(11,11),1.5)
        s1=cv2.GaussianBlur(a**2,(11,11),1.5)-m1**2; s2=cv2.GaussianBlur(b**2,(11,11),1.5)-m2**2
        s12=cv2.GaussianBlur(a*b,(11,11),1.5)-m1*m2
        v.append(float(((2*m1*m2+C1)*(2*s12+C2)/((m1**2+m2**2+C1)*(s1+s2+C2))).mean()))
    return float(np.mean(v))

fnames=sorted(f for f in os.listdir(INP_DIR) if f.endswith('.png'))
gt_ns =sorted(f for f in os.listdir(GT_DIR)  if f.endswith('.png'))
im=  {os.path.splitext(f)[0]:f for f in fnames}
gm=  {os.path.splitext(f)[0]:f for f in gt_ns}
stems=sorted(set(im)&set(gm)); N=len(stems)

print(f"Loading original model ...")
net_o = SYEISPNet(channels=12,rep_scale=4).eval()
sd = torch.load(os.path.join(ARCHIVE,'model_best.pkl'),map_location='cpu',weights_only=False)
net_o.load_state_dict(sd)
net_s = net_o.slim().eval()

csv_path=os.path.join(RESULTS_DIR,'per_image_results.csv')
cols=['model','image_name','psnr','ssim','mse','mae','rmse','h2d_time_ms',
      'preprocess_time_ms','inference_time_ms','postprocess_time_ms','d2h_time_ms','total_latency_ms']

# Load existing TFLite rows
existing=[]
if os.path.exists(csv_path):
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            existing.append(row)

def run_pt(net, model_key, label):
    net=net.to(DEVICE)
    rows=[]; print(f"\n[{label}] {N} images ...")
    with torch.no_grad():
        for i,stem in enumerate(stems):
            raw=np.array(Image.open(os.path.join(INP_DIR,im[stem])))
            gt =np.array(Image.open(os.path.join(GT_DIR,gm[stem])).convert('RGB')).astype(np.float32)/255.
            rggb=bayer2rggb(raw)
            e=[torch.cuda.Event(enable_timing=True) for _ in range(10)]
            t0=time.perf_counter()
            e[0].record(); inp=torch.from_numpy(rggb).unsqueeze(0).to(DEVICE); e[1].record()
            e[2].record(); e[3].record()
            e[4].record(); out_gpu=net(inp); e[5].record()
            e[6].record(); out_c=out_gpu.clamp(0,1); e[7].record()
            e[8].record(); out_cpu=out_c.squeeze().permute(1,2,0).cpu().numpy(); e[9].record()
            torch.cuda.synchronize(); t1=time.perf_counter()
            h2d=e[0].elapsed_time(e[1]); pre=e[2].elapsed_time(e[3])
            inf=e[4].elapsed_time(e[5]); pst=e[6].elapsed_time(e[7]); d2h=e[8].elapsed_time(e[9])
            diff=out_cpu.astype(np.float64)-gt.astype(np.float64)
            mse=float((diff**2).mean()); mae=float(np.abs(diff).mean())
            psnr=compute_psnr(out_cpu,gt); ssim=compute_ssim(out_cpu,gt)
            if i>=WARMUP:
                rows.append({'model':model_key,'image_name':stem,
                    'psnr':psnr,'ssim':ssim,'mse':mse,'mae':mae,'rmse':math.sqrt(mse),
                    'h2d_time_ms':h2d,'preprocess_time_ms':pre,'inference_time_ms':inf,
                    'postprocess_time_ms':pst,'d2h_time_ms':d2h,'total_latency_ms':(t1-t0)*1000.})
            if (i+1)%500==0:
                print(f"  {i+1}/{N} PSNR={psnr:.2f} inf={inf:.2f}ms")
    psnrs=[r['psnr'] for r in rows]; times=[r['total_latency_ms'] for r in rows]
    ssims=[r['ssim'] for r in rows]
    print(f"  -> PSNR={statistics.mean(psnrs):.4f} SSIM={statistics.mean(ssims):.4f} lat={statistics.mean(times):.3f}ms FPS={1000/statistics.mean(times):.2f}")
    return rows

rows_o=run_pt(net_o,'original_pytorch','Original PyTorch GPU')
rows_s=run_pt(net_s,'slim_pytorch',    'Slim PyTorch GPU')

with open(csv_path,'w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=cols,extrasaction='ignore')
    w.writeheader()
    for r in rows_o: w.writerow(r)
    for r in rows_s: w.writerow(r)
    for r in existing: w.writerow(r)
print(f"\nWritten {len(rows_o)+len(rows_s)+len(existing)} rows to {csv_path}")

# Update summary CSV with PyTorch stats
def summarize(rows,label,size_kb,params):
    psnrs=[r['psnr'] for r in rows]; ssims=[r['ssim'] for r in rows]
    mses=[r['mse'] for r in rows]; maes=[r['mae'] for r in rows]; rmses=[r['rmse'] for r in rows]
    times=[r['total_latency_ms'] for r in rows]; infs=[r['inference_time_ms'] for r in rows]
    best_pi=rows[psnrs.index(max(psnrs))]; worst_pi=rows[psnrs.index(min(psnrs))]
    return {'label':label,'size_kb':size_kb,'params':params,
        'avg_psnr':statistics.mean(psnrs),'avg_ssim':statistics.mean(ssims),
        'avg_mse':statistics.mean(mses),'avg_mae':statistics.mean(maes),'avg_rmse':statistics.mean(rmses),
        'lat_avg':statistics.mean(times),'lat_min':min(times),'lat_max':max(times),
        'lat_median':statistics.median(times),'lat_std':statistics.stdev(times),
        'fps':1000/statistics.mean(times),'total_images':len(rows),'total_time_s':sum(times)/1000.,
        'avg_sm_util':'N/A','peak_sm_util':'N/A','avg_mem_mb':'N/A','peak_mem_mb':'N/A',
        'avg_power_w':'N/A','avg_temp_c':'N/A'}

import os
orig_size=os.path.getsize(os.path.join(ARCHIVE,'model_best.pkl'))/1024
slim_size=os.path.getsize(os.path.join(ARCHIVE,'slim_model.pt'))/1024
S_O=summarize(rows_o,'Original PyTorch GPU',orig_size,110984)
S_S=summarize(rows_s,'Slim PyTorch GPU',slim_size,5640)

sum_path=os.path.join(RESULTS_DIR,'summary_metrics.csv')
existing_sum=[]
if os.path.exists(sum_path):
    with open(sum_path) as f:
        for row in csv.DictReader(f):
            if 'PyTorch' not in row.get('label',''):
                existing_sum.append(row)

sum_fields=['label','size_kb','params','avg_psnr','avg_ssim','avg_mse','avg_mae','avg_rmse',
            'lat_avg','lat_min','lat_max','lat_median','lat_std','fps',
            'total_images','total_time_s','avg_sm_util','peak_sm_util',
            'avg_mem_mb','peak_mem_mb','avg_power_w','avg_temp_c']

def fv(v):
    if isinstance(v,float) and math.isnan(v): return 'N/A'
    if isinstance(v,float): return f'{v:.6f}'
    return str(v)

with open(sum_path,'w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=sum_fields,extrasaction='ignore')
    w.writeheader()
    for s in [S_O,S_S]: w.writerow({k:fv(s.get(k,'N/A')) for k in sum_fields})
    for r in existing_sum: w.writerow(r)
print(f"Updated: {sum_path}")
