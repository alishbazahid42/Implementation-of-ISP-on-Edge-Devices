import os
import sys
import argparse
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from PIL import Image, ImageOps

# ---------------------------------------------------------
# PyTorch SSIM Implementation (Research Standard)
# ---------------------------------------------------------
def gaussian(window_size, sigma):
    gauss = torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel):
    mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size//2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size//2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size//2, groups=channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

def calculate_ssim(img1, img2):
    img1_t = torch.from_numpy(img1).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    img2_t = torch.from_numpy(img2).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    
    channel = img1_t.size(1)
    window_size = 11
    window = create_window(window_size, channel)
    window = window.type_as(img1_t)
    
    return _ssim(img1_t, img2_t, window, window_size, channel).item()

# ---------------------------------------------------------
# PSNR Implementation
# ---------------------------------------------------------
def calculate_psnr(img1, img2):
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

# ---------------------------------------------------------
# MAIN VALIDATION SCRIPT
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Calculate PSNR & SSIM metrics between AI Output and Ground Truth")
    parser.add_argument("output_img", help="Path to the processed output image (.png)")
    parser.add_argument("ground_truth", help="Path to the Ground Truth image (.jpeg/.jpg)")
    args = parser.parse_args()

    if not os.path.exists(args.output_img):
        print(f"ERROR: Output image '{args.output_img}' not found.")
        sys.exit(1)
    if not os.path.exists(args.ground_truth):
        print(f"ERROR: Ground Truth image '{args.ground_truth}' not found.")
        sys.exit(1)

    print(f"\nEvaluating: '{os.path.basename(args.output_img)}' against Ground Truth: '{os.path.basename(args.ground_truth)}'")

    # Load Ground Truth and ensure physical EXIF rotation is applied
    gt_img = Image.open(args.ground_truth).convert("RGB")
    gt_img = ImageOps.exif_transpose(gt_img)
    gt_np = np.array(gt_img)

    # Load AI Output and perfectly align resolution to prevent broadcast mismatch crashes
    out_img = Image.open(args.output_img).convert("RGB")
    if out_img.size != gt_img.size:
        print(f"  -> Resizing AI Output from {out_img.size} to {gt_img.size} for pixel-perfect alignment...")
        out_img = out_img.resize((gt_img.width, gt_img.height), Image.Resampling.LANCZOS)
    
    out_np = np.array(out_img)

    print("\nCalculating mathematical metrics...")
    psnr_val = calculate_psnr(gt_np, out_np)
    ssim_val = calculate_ssim(gt_np, out_np)

    print("-" * 50)
    print("FINAL RESULTS:")
    print(f"PSNR: {psnr_val:.2f} dB")
    print(f"SSIM: {ssim_val:.4f}")
    print("-" * 50 + "\n")

if __name__ == "__main__":
    main()
