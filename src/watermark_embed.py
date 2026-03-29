"""
Step 2: Embed invisible watermarks and measure quality
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
ORIGINAL_DIR   = "dataset/original"
WATERMARKED_DIR = "dataset/watermarked"
RESULTS_DIR    = "results"
WATERMARK_MSG  = "10101010110011001111000010101010"
METHOD         = "dwtDct"

def embed_single(img_path, out_path):
    """Embed watermark into one image."""
    img_np = np.array(
        Image.open(img_path).convert("RGB")
    )
    encoder = WatermarkEncoder()
    encoder.set_watermark('bits', WATERMARK_MSG)
    encoded = encoder.encode(img_np, METHOD)
    Image.fromarray(encoded).save(out_path)
    return img_np, encoded

def detect_single(img_path):
    """Detect watermark from one image."""
    img_np = np.array(
        Image.open(img_path).convert("RGB")
    )
    decoder = WatermarkDecoder('bits', 32)
    return decoder.decode(img_np, METHOD)

def bit_accuracy(detected, original=WATERMARK_MSG):
    """Calculate bit accuracy."""
    if detected is None:
        return 0.0
    return sum(d == o for d, o in
               zip(detected, original)) / len(original)

def embed_all():
    os.makedirs(WATERMARKED_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    files = [f for f in os.listdir(ORIGINAL_DIR)
             if f.endswith(".png")]

    psnr_list, ssim_list, acc_list = [], [], []

    for fname in tqdm(files, desc="Embedding watermarks"):
        orig_path = os.path.join(ORIGINAL_DIR, fname)
        wm_path   = os.path.join(WATERMARKED_DIR, fname)

        orig_np, wm_np = embed_single(orig_path, wm_path)

        # Quality metrics
        psnr = peak_signal_noise_ratio(orig_np, wm_np)
        ssim = structural_similarity(
            orig_np, wm_np, channel_axis=2
        )

        # Detection accuracy
        detected = detect_single(wm_path)
        acc = bit_accuracy(detected)

        psnr_list.append(psnr)
        ssim_list.append(ssim)
        acc_list.append(acc)

    # Save summary
    summary = {
        "total_images":   len(files),
        "avg_psnr":       float(np.mean(psnr_list)),
        "avg_ssim":       float(np.mean(ssim_list)),
        "avg_bit_accuracy": float(np.mean(acc_list)),
    }

    out_file = os.path.join(
        RESULTS_DIR, "embedding_results.json"
    )
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n── Embedding Results ──────────────────")
    print(f"Total images  : {summary['total_images']}")
    print(f"Avg PSNR      : {summary['avg_psnr']:.2f} dB")
    print(f"Avg SSIM      : {summary['avg_ssim']:.4f}")
    print(f"Avg Bit Acc   : "
          f"{summary['avg_bit_accuracy']*100:.1f}%")
    print(f"Results saved : {out_file}")

if __name__ == "__main__":
    embed_all()