import os
import sys
import io
import time
import argparse
import numpy as np
import rawpy
import torch
from PIL import Image

# ---------------------------------------------------------
# DEPENDENCY CHECK: Ensure original SYENET is available
# ---------------------------------------------------------
try:
    from syenet.model.isp import SYEISPNetS
except ImportError:
    print("ERROR: Could not import SYEISPNetS.")
    print("Please ensure you have cloned the official SYENET repository next to this script.")
    print("Run: git clone https://github.com/sanechips-multimedia/syenet.git")
    sys.exit(1)

# ---------------------------------------------------------
# CORE ALGORITHM: Zero-Shot Sensor Domain Adaptation
# ---------------------------------------------------------
def format_bayer_data(bayer):
    """
    Translates raw data from a 14-bit Google Pixel 7a sensor into a 
    12-bit Mediatek/Fujifilm profile that the SYENET AI expects.
    """
    # Extract the RGGB Bayer pattern channels
    R = bayer[0::2, 0::2]
    G1 = bayer[0::2, 1::2]
    G2 = bayer[1::2, 0::2]
    B = bayer[1::2, 1::2]

    # Global exposure and color white-balance multipliers tailored for Pixel 7a
    global_exposure = 1.20 
    r_nudge = 1.118
    g_nudge = 1.042
    b_nudge = 0.943
    
    r_gain = global_exposure * r_nudge
    g_gain = global_exposure * g_nudge
    b_gain = global_exposure * b_nudge

    # Target mathematical constraints for SYENET (trained on MAI dataset)
    target_black = 344.0
    target_white = 4095.0
    
    # Scale from Pixel 7a 14-bit max (16383) to SYENET 12-bit max (4095)
    scale_factor = (target_white - target_black) / 16383.0

    # Apply mathematical scaling, color gains, and inject the +344 Black Level noise floor
    R_scaled = R * r_gain * scale_factor + target_black
    G1_scaled = G1 * g_gain * scale_factor + target_black
    G2_scaled = G2 * g_gain * scale_factor + target_black
    B_scaled = B * b_gain * scale_factor + target_black

    # Reconstruct the Bayer pattern
    bayer_12bit = np.zeros_like(bayer)
    bayer_12bit[0::2, 0::2] = R_scaled
    bayer_12bit[0::2, 1::2] = G1_scaled
    bayer_12bit[1::2, 0::2] = G2_scaled
    bayer_12bit[1::2, 1::2] = B_scaled

    # Mathematically clip to strictly enforce 12-bit limits and package into 16-bit PNG container
    bayer_16bit_png = np.clip(np.round(bayer_12bit), 0, 4095).astype(np.uint16)
    return bayer_16bit_png

def bayer2rggb(img_bayer):
    """Reshapes the flat 2D Bayer array into a 4-channel RGGB tensor for AI Processing."""
    h, w = img_bayer.shape
    img_bayer = img_bayer.reshape(h // 2, 2, w // 2, 2)
    img_bayer = img_bayer.transpose([1, 3, 0, 2]).reshape([-1, h // 2, w // 2])
    return img_bayer

def apply_orientation(img, flip_val):
    """Mathematically rotates the output based on physical gravity sensor flags."""
    if flip_val == 3:
        return img.transpose(Image.Transpose.ROTATE_180)
    elif flip_val == 5:
        return img.transpose(Image.Transpose.ROTATE_90).transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    elif flip_val == 6:
        return img.transpose(Image.Transpose.ROTATE_270)
    elif flip_val == 8:
        return img.transpose(Image.Transpose.ROTATE_90)
    elif flip_val == 2:
        return img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    elif flip_val == 4:
        return img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    return img

# ---------------------------------------------------------
# MAIN PIPELINE EXECUTION
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Zero-Shot Domain Adaptation and Inference for SYENET")
    parser.add_argument("input_dng", help="Path to the input raw .dng file (e.g., Pixel 7a image)")
    parser.add_argument("--weights", default="weights/model_best_slim.pkl", help="Path to SYENET weights")
    args = parser.parse_args()

    input_path = args.input_dng
    if not os.path.exists(input_path):
        print(f"ERROR: Input file {input_path} not found.")
        sys.exit(1)

    print(f"\n[1/4] Loading RAW Image and Extracting Metadata from '{os.path.basename(input_path)}'...")
    with rawpy.imread(input_path) as raw:
        bayer_full = raw.raw_image_visible.astype(np.float32)
        flip_val = raw.sizes.flip
        print(f"      Resolution: {bayer_full.shape[1]}x{bayer_full.shape[0]}")
        print(f"      Hardware Rotation Flag (flip_val): {flip_val}")

    # Ensure even dimensions to prevent Bayer pattern splitting
    h, w = bayer_full.shape
    bayer_full = bayer_full[:h - (h % 2), :w - (w % 2)]

    print("\n[2/4] Applying Sensor Domain Adaptation...")
    # Translate Pixel 7a physics to SYENET physics
    formatted_bayer = format_bayer_data(bayer_full)
    
    # Prepare PyTorch tensor
    raw_np = bayer2rggb(formatted_bayer)
    raw_np = raw_np.astype(np.float32) / 4095.0
    input_tensor = torch.from_numpy(raw_np).unsqueeze(0)

    print("\n[3/4] Loading AI Model & Running Inference...")
    model = SYEISPNetS(channels=12)
    if not os.path.exists(args.weights):
        print(f"ERROR: Weights file not found at {args.weights}")
        sys.exit(1)
        
    state_dict = torch.load(args.weights, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()

    start_time = time.time()
    with torch.no_grad():
        output_tensor = model(input_tensor)
    inf_time = time.time() - start_time
    print(f"      Inference completed in {inf_time:.2f} seconds.")

    print("\n[4/4] Applying Post-Processing & Hardware Orientation...")
    out_img_arr = output_tensor.squeeze(0).permute(1, 2, 0).numpy()
    out_img_arr = np.clip(out_img_arr, 0.0, 1.0)
    out_img_arr = (out_img_arr * 255.0).astype(np.uint8)

    final_image = Image.fromarray(out_img_arr)
    if flip_val != 0:
        final_image = apply_orientation(final_image, flip_val)

    # Save Output
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    output_filename = f"{base_name}_processed.png"
    final_image.save(output_filename, format="PNG")
    print(f"\nSUCCESS! Fully processed image saved as: {output_filename}\n")

if __name__ == "__main__":
    main()
