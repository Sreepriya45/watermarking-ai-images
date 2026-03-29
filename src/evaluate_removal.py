
"""
Evaluate how well ViT removed the watermarks
"""
import numpy as np
from PIL import Image
from imwatermark import WatermarkDecoder
import os
import json

WATERMARKED_DIR = "data/watermarked"
CLEANED_DIR     = "data/cleaned"
ORIGINAL_DIR    = "data/original"
RESULTS_DIR     = "results/tables"
WATERMARK_MSG   = "10101010110011001111000010101010"
WATERMARK_BITS  = 32
METHOD          = "dwtDct"

def detect(img_path):
    img = Image.open(img_path).convert("RGB")
    # Resize to minimum required size for decoder
    if img.size[0] < 256 or img.size[1] < 256:
        img = img.resize((256, 256), Image.LANCZOS)
    img_np = np.array(img)
    decoder = WatermarkDecoder('bits', WATERMARK_BITS)
    return decoder.decode(img_np, METHOD)

def bit_acc(detected, original=WATERMARK_MSG):
    if detected is None:
        return 0.0
    original_bits = [b == '1' for b in original]
    if hasattr(detected, 'tolist'):
        detected_bits = detected.tolist()
    else:
        detected_bits = list(detected)
    matches = sum(
        bool(d) == bool(o)
        for d, o in zip(detected_bits, original_bits)
    )
    return matches / len(original_bits)

from skimage.metrics import (
    peak_signal_noise_ratio,
    structural_similarity
)

files = [f for f in os.listdir(CLEANED_DIR)
         if f.endswith(".png")]

wm_acc_list    = []
clean_acc_list = []
psnr_list      = []
ssim_list      = []

for fname in files:
    wm_path    = os.path.join(WATERMARKED_DIR, fname)
    clean_path = os.path.join(CLEANED_DIR, fname)
    orig_path  = os.path.join(ORIGINAL_DIR, fname)

    # Bit accuracy BEFORE removal (watermarked)
    wm_detected = detect(wm_path)
    wm_acc      = bit_acc(wm_detected)

    # Bit accuracy AFTER removal (cleaned by ViT)
    cl_detected = detect(clean_path)
    cl_acc      = bit_acc(cl_detected)

    # Image quality after removal
    orig_np  = np.array(
        Image.open(orig_path).convert("RGB")
        .resize((224, 224))
    )
    clean_np = np.array(
        Image.open(clean_path).convert("RGB")
    )

    psnr = peak_signal_noise_ratio(orig_np, clean_np)
    ssim = structural_similarity(
        orig_np, clean_np, channel_axis=2
    )

    wm_acc_list.append(wm_acc)
    clean_acc_list.append(cl_acc)
    psnr_list.append(psnr)
    ssim_list.append(ssim)

# ── Print Results ──────────────────────────────────
print("\n── ViT Removal Evaluation ─────────────────")
print(f"Images evaluated       : {len(files)}")
print(f"Bit Acc (watermarked)  : "
      f"{np.mean(wm_acc_list)*100:.1f}%")
print(f"Bit Acc (after removal): "
      f"{np.mean(clean_acc_list)*100:.1f}%")
print(f"Watermark reduction    : "
      f"{(np.mean(wm_acc_list) - np.mean(clean_acc_list))*100:.1f}%")
print(f"Avg PSNR after removal : "
      f"{np.mean(psnr_list):.2f} dB")
print(f"Avg SSIM after removal : "
      f"{np.mean(ssim_list):.4f}")

# ── Save ───────────────────────────────────────────
results = {
    "images_evaluated":       len(files),
    "bit_acc_watermarked":    round(float(np.mean(wm_acc_list)), 4),
    "bit_acc_after_removal":  round(float(np.mean(clean_acc_list)), 4),
    "watermark_reduction_pct": round(float(
        (np.mean(wm_acc_list) - np.mean(clean_acc_list)) * 100
    ), 2),
    "avg_psnr_after_removal": round(float(np.mean(psnr_list)), 2),
    "avg_ssim_after_removal": round(float(np.mean(ssim_list)), 4),
}

out_file = os.path.join(
    RESULTS_DIR, "vit_removal_results.json"
)
with open(out_file, "w") as f:
    json.dump(results, f, indent=2)

print(f"\nResults saved to: {out_file}")