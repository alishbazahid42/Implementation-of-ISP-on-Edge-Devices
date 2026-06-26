import os
import sys
import numpy as np
import rawpy
import torch
from PIL import Image

try:
    from syenet.model.isp import SYEISPNetS
except ImportError:
    print("ERROR: SYEISPNetS missing.")
    sys.exit(1)

def bayer2rggb(img_bayer):
    h, w = img_bayer.shape
    return img_bayer.reshape(h // 2, 2, w // 2, 2).transpose([1, 3, 0, 2]).reshape([-1, h // 2, w // 2])

def apply_orientation(img, flip_val):
    if flip_val == 3: return img.transpose(Image.Transpose.ROTATE_180)
    elif flip_val == 5: return img.transpose(Image.Transpose.ROTATE_90).transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    elif flip_val == 6: return img.transpose(Image.Transpose.ROTATE_270)
    elif flip_val == 8: return img.transpose(Image.Transpose.ROTATE_90)
    elif flip_val == 2: return img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    elif flip_val == 4: return img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    return img

def run_inference(bayer_12bit, flip_val, output_name):
    # Prepare PyTorch tensor
    raw_np = bayer2rggb(bayer_12bit)
    raw_np = raw_np.astype(np.float32) / 4095.0
    input_tensor = torch.from_numpy(raw_np).unsqueeze(0)

    model = SYEISPNetS(channels=12)
    model.load_state_dict(torch.load("weights/model_best_slim.pkl", map_location='cpu'))
    model.eval()

    with torch.no_grad():
        output_tensor = model(input_tensor)

    out_img_arr = output_tensor.squeeze(0).permute(1, 2, 0).numpy()
    out_img_arr = np.clip(out_img_arr, 0.0, 1.0)
    out_img_arr = (out_img_arr * 255.0).astype(np.uint8)

    final_image = Image.fromarray(out_img_arr)
    if flip_val != 0:
        final_image = apply_orientation(final_image, flip_val)

    final_image.save(output_name, format="PNG")
    print(f"Generated: {output_name}")

def main():
    input_path = r"C:\Users\Mc\Desktop\SYENET\RAW-Image\Custom-Dataset\Pixel-6-raw\indoor_1.dng"
    print(f"Loading '{os.path.basename(input_path)}'...")
    with rawpy.imread(input_path) as raw:
        bayer_full = raw.raw_image_visible.astype(np.float32)
        flip_val = raw.sizes.flip

    h, w = bayer_full.shape
    bayer_full = bayer_full[:h - (h % 2), :w - (w % 2)]

    os.makedirs("examples", exist_ok=True)

    # ---------------------------------------------------------
    # VARIATION 2: Just Bit Scaling (No Black Level, No Color)
    # ---------------------------------------------------------
    scale_factor = 4095.0 / 16383.0
    bayer_v2 = bayer_full * scale_factor
    bayer_v2_16bit = np.clip(np.round(bayer_v2), 0, 4095).astype(np.uint16)
    run_inference(bayer_v2_16bit, flip_val, "examples/v2_scaled_no_bl.png")

    # ---------------------------------------------------------
    # VARIATION 3: Bit Scaling + +344 Black Level (No Color)
    # ---------------------------------------------------------
    bayer_v3 = (bayer_full * scale_factor) + 344.0
    bayer_v3_16bit = np.clip(np.round(bayer_v3), 0, 4095).astype(np.uint16)
    run_inference(bayer_v3_16bit, flip_val, "examples/v3_scaled_bl_no_color.png")

    print("\nSUCCESS! Variations generated.")

if __name__ == "__main__":
    main()
