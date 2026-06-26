"""Repack the unzipped PyTorch checkpoint back into a valid .pt zip file."""
import zipfile, os, shutil

root   = r'C:\Users\aujla\Desktop\archive'
out_pt = os.path.join(root, 'slim_model.pt')

with zipfile.ZipFile(out_pt, 'w', compression=zipfile.ZIP_STORED) as z:
    z.write(os.path.join(root, 'data.pkl'),  arcname='archive/data.pkl')
    z.write(os.path.join(root, 'version'),   arcname='archive/version')
    for f in sorted(os.listdir(os.path.join(root, 'data'))):
        z.write(os.path.join(root, 'data', f), arcname=f'archive/data/{f}')

size = os.path.getsize(out_pt) / 1024
print(f"Written: {out_pt}  ({size:.1f} KB)")

# Quick verify: load with torch
import torch
sd = torch.load(out_pt, map_location='cpu', weights_only=False)
print(f"State dict keys ({len(sd)}): {list(sd.keys())}")
total = sum(v.numel() for v in sd.values())
print(f"Total parameters: {total:,}")
