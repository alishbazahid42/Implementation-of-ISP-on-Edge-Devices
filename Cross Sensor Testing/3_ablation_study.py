import os
import sys
import time
import argparse
import numpy as np
import rawpy
import torch
from PIL import Image

try:
    from syenet.model.isp import SYEISPNetS
except ImportError:
    print("ERROR: SYEISPNetS missing. Ensure syenet/ is cloned here.")
    sys.exit(1)

def bayer2rggb(img_bayer):
    h, w = img_bayer.shape
    img_bayer = img_bayer.reshape(h // 2, 2, w // 2, 2)
    img_bayer = img_bayer.transpose([1, 3, 0, 2]).reshape([-1, h // 2, w // 2])
    return img_bayer

def apply_orientation(img, flip_val):
    if flip_val == 3: return img.transpose(Image.Transpose.ROTATE_180)
    elif flip_val == 5: return img.transpose(Image.Transpose.ROTATE_90).transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    elif flip_val == 6: return img.transpose(Image.Transpose.ROTATE_270)
    elif flip_val == 8: return img.transpose(Image.Transpose.ROTATE_90)
    elif flip_val == 2: return img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    elif flip_val == 4: return img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    return img

def main():
    parser = argparse.ArgumentParser(description="Ablation Study: Inference without Domain Adaptation")
    parser.add_argument("input_dng", help="Path to input .dng file")
    parser.add_argument("--weights", default="weights/model_best_slim.pkl", help="Path to weights")
    args = parser.parse_args()

    print(f"\n[ABLATION STUDY] Loading '{os.path.basename(args.input_dng)}'...")
    with rawpy.imread(args.input_dng) as raw:
        bayer_full = raw.raw_image_visible.astype(np.float32)
        flip_val = raw.sizes.flip

    h, w = bayer_full.shape
    bayer_full = bayer_full[:h - (h % 2), :w - (w % 2)]

    print("\n[WARNING] Bypassing Sensor Domain Adaptation!")
    print("          Passing 14-bit Pixel 7a data directly into 12-bit AI...")
    
    # INTENTIONAL FAILURE: We do NOT scale bits, do NOT add +344 black level, do NOT fix colors.
    raw_np = bayer2rggb(bayer_full)
    
    # The AI expects data normalized to 4095. 
    # Because Pixel 7a data goes up to 16383, dividing by 4095 will cause values > 1.0 (Overflow)
    raw_np = raw_np.astype(np.float32) / 4095.0 
    
    input_tensor = torch.from_numpy(raw_np).unsqueeze(0)

    print("\nRunning Inference...")
    model = SYEISPNetS(channels=12)
    model.load_state_dict(torch.load(args.weights, map_location='cpu'))
    model.eval()

    with torch.no_grad():
        output_tensor = model(input_tensor)

    out_img_arr = output_tensor.squeeze(0).permute(1, 2, 0).numpy()
    
    # The overflowed mathematical values will cause intense highlight clipping and color corruption
    out_img_arr = np.clip(out_img_arr, 0.0, 1.0)
    out_img_arr = (out_img_arr * 255.0).astype(np.uint8)

    final_image = Image.fromarray(out_img_arr)
    if flip_val != 0:
        final_image = apply_orientation(final_image, flip_val)

    os.makedirs("examples", exist_ok=True)
    output_filename = "examples/2_bad_preprocessing.png"
    final_image.save(output_filename, format="PNG")
    print(f"\nSUCCESS! Corrupted Ablation Image saved to: {output_filename}\n")

if __name__ == "__main__":
    main()
