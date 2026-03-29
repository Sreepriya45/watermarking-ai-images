"""
Step 2: Embed invisible watermarks into AI-generated 
images and measure quality
"""
import numpy as np
from PIL import Image
from imwatermark import WatermarkEncoder, WatermarkDecoder
from skimage.metrics import (peak_signal_noise_ratio,
                              structural_similarity)
import os
import json
from tqdm import tqdm

# ── Config ─────────────────────────────────────────
ORIGINAL_DIR    = "data/original"
WATERMARKED_DIR = "data/watermarked"
RESULTS_DIR     = "results/tables"
# WITH THIS
WATERMARK_BITS  = 32
WATERMARK_MSG   = "10101010110011001111000010101010"  # exactly 32 chars
METHOD          = "dwtDct"



def embed_single(img_path, out_path):
    """Embed watermark into one image."""
    img = Image.open(img_path).convert("RGB")
    img_np = np.array(img)
    
    # Resize to minimum required size
    if img_np.shape[0] < 256 or img_np.shape[1] < 256:
        img = img.resize((256, 256))
        img_np = np.array(img)
    
    encoder = WatermarkEncoder()
    encoder.set_watermark('bits', WATERMARK_MSG)
    encoded = encoder.encode(img_np, METHOD)
    Image.fromarray(encoded).save(out_path)
    return img_np, encoded

def detect_single(img_path):
    """Detect watermark from one image."""
    img = Image.open(img_path).convert("RGB")
    img_np = np.array(img)
    decoder = WatermarkDecoder('bits', WATERMARK_BITS)
    decoded = decoder.decode(img_np, METHOD)
    return decoded

def bit_accuracy(detected, original=WATERMARK_MSG):
    """Calculate what % of watermark bits are correct."""
    if detected is None:
        return 0.0
    
    # Convert original string to boolean list
    original_bits = [b == '1' for b in original]
    
    # Handle numpy boolean array
    if hasattr(detected, 'tolist'):
        detected_bits = detected.tolist()
    else:
        detected_bits = list(detected)
    
    # Compare booleans directly
    matches = sum(
        bool(d) == bool(o)
        for d, o in zip(detected_bits, original_bits)
    )
    return matches / len(original_bits)

def embed_all():
    os.makedirs(WATERMARKED_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    files = [f for f in os.listdir(ORIGINAL_DIR)
             if f.endswith(".png")]
    
    print(f"Found {len(files)} images to watermark...")

    psnr_list = []
    ssim_list = []
    acc_list  = []
    per_image = []

    for fname in tqdm(files, desc="Embedding watermarks"):
        orig_path = os.path.join(ORIGINAL_DIR, fname)
        wm_path   = os.path.join(WATERMARKED_DIR, fname)

        # Embed watermark
        orig_np, wm_np = embed_single(orig_path, wm_path)

        # Measure image quality
        psnr = peak_signal_noise_ratio(orig_np, wm_np)
        ssim = structural_similarity(
            orig_np, wm_np, channel_axis=2
        )

        # Measure detection accuracy
        detected = detect_single(wm_path)
        acc = bit_accuracy(detected)

        psnr_list.append(psnr)
        ssim_list.append(ssim)
        acc_list.append(acc)

        per_image.append({
            "image":        fname,
            "psnr":         round(float(psnr), 2),
            "ssim":         round(float(ssim), 4),
            "bit_accuracy": round(float(acc), 4),
        })

        print(f"  {fname} — PSNR: {psnr:.2f} dB  "
              f"SSIM: {ssim:.4f}  "
              f"Bit Acc: {acc*100:.1f}%")

    # ── Summary ────────────────────────────────────
    summary = {
        "total_images":     len(files),
        "avg_psnr_db":      round(float(np.mean(psnr_list)), 2),
        "avg_ssim":         round(float(np.mean(ssim_list)), 4),
        "avg_bit_accuracy": round(float(np.mean(acc_list)), 4),
        "per_image":        per_image
    }

    out_file = os.path.join(
        RESULTS_DIR, "embedding_results.json"
    )
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)

    # ── Print Summary ──────────────────────────────
    print("\n── Embedding Summary ──────────────────────")
    print(f"Total images    : {summary['total_images']}")
    print(f"Avg PSNR        : {summary['avg_psnr_db']} dB")
    print(f"Avg SSIM        : {summary['avg_ssim']}")
    print(f"Avg Bit Accuracy: "
          f"{summary['avg_bit_accuracy']*100:.1f}%")
    print(f"\nResults saved to: {out_file}")
    print("\nWatermarked images saved to: data/watermarked/")

if __name__ == "__main__":
    embed_all()
    
def test_one_image():
    """Quick test on a single image."""
    test_img = os.path.join(
        ORIGINAL_DIR,
        os.listdir(ORIGINAL_DIR)[0]
    )
    test_out = "data/watermarked/test_debug.png"
    
    print("\n── Debug Test ─────────────────────────")
    print(f"Testing on: {test_img}")
    
    # Embed
    orig_np, wm_np = embed_single(test_img, test_out)
    print(f"Embedded successfully")
    
    # Detect
    detected = detect_single(test_out)
    print(f"Detected type  : {type(detected)}")
    print(f"Detected value : {detected}")
    print(f"Original msg   : {WATERMARK_MSG}")
    
    # Accuracy
    acc = bit_accuracy(detected)
    print(f"Bit Accuracy   : {acc*100:.1f}%")

if __name__ == "__main__":
    test_one_image()  # Test first
    # embed_all()     # Comment this out for now


## What This Script Does

## data/original/img_00_00.png
##         ↓
## Embed invisible watermark (dwtDct method)
##         ↓
## data/watermarked/img_00_00.png

## Then measures:
#- PSNR  → should be above 35 dB (good quality)
#- SSIM  → should be above 0.95 (visually similar)
#- Bit Accuracy → should be 100% (watermark detected)