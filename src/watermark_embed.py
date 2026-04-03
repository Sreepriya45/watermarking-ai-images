"""
Step 2: Embed invisible watermarks into AI-generated
images and measure quality — comparing dwtDct vs dwtDctSvd
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
ORIGINAL_DIR   = "data/original"
RESULTS_DIR    = "results/tables"
WATERMARK_BITS = 32
WATERMARK_MSG  = "10101010110011001111000010101010"  # exactly 32 chars
METHODS        = ["dwtDct", "dwtDctSvd"]

# dwtDct remains the primary dir for downstream scripts
WATERMARKED_DIRS = {
    "dwtDct":    "data/watermarked",
    "dwtDctSvd": "data/watermarked_dwtDctSvd",
}


def embed_single(img_path, out_path, method):
    """Embed watermark into one image."""
    img = Image.open(img_path).convert("RGB")
    img_np = np.array(img)

    if img_np.shape[0] < 256 or img_np.shape[1] < 256:
        img = img.resize((256, 256))
        img_np = np.array(img)

    encoder = WatermarkEncoder()
    encoder.set_watermark('bits', WATERMARK_MSG)
    encoded = encoder.encode(img_np, method)
    Image.fromarray(encoded).save(out_path)
    return img_np, encoded


def detect_single(img_path, method):
    """Detect watermark from one image."""
    img = Image.open(img_path).convert("RGB")
    img_np = np.array(img)
    decoder = WatermarkDecoder('bits', WATERMARK_BITS)
    return decoder.decode(img_np, method)


def bit_accuracy(detected, original=WATERMARK_MSG):
    """Calculate what % of watermark bits are correct."""
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


def embed_all(method):
    """Run embedding pipeline for one method. Returns summary dict."""
    watermarked_dir = WATERMARKED_DIRS[method]
    os.makedirs(watermarked_dir, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    files = [f for f in os.listdir(ORIGINAL_DIR)
             if f.endswith(".png")]

    print(f"\n[{method}] Found {len(files)} images...")

    psnr_list, ssim_list, acc_list, per_image = [], [], [], []

    for fname in tqdm(files, desc=f"{method}"):
        orig_path = os.path.join(ORIGINAL_DIR, fname)
        wm_path   = os.path.join(watermarked_dir, fname)

        orig_np, wm_np = embed_single(orig_path, wm_path, method)

        psnr = peak_signal_noise_ratio(orig_np, wm_np)
        ssim = structural_similarity(
            orig_np, wm_np, channel_axis=2
        )
        detected = detect_single(wm_path, method)
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

    summary = {
        "method":           method,
        "total_images":     len(files),
        "avg_psnr_db":      round(float(np.mean(psnr_list)), 2),
        "avg_ssim":         round(float(np.mean(ssim_list)), 4),
        "avg_bit_accuracy": round(float(np.mean(acc_list)), 4),
        "per_image":        per_image,
    }

    out_file = os.path.join(
        RESULTS_DIR, f"embedding_{method}.json"
    )
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"  Avg PSNR        : {summary['avg_psnr_db']} dB")
    print(f"  Avg SSIM        : {summary['avg_ssim']}")
    print(f"  Avg Bit Accuracy: "
          f"{summary['avg_bit_accuracy']*100:.1f}%")

    return summary


if __name__ == "__main__":
    all_results = []
    for method in METHODS:
        all_results.append(embed_all(method))

    # ── Comparison Table ───────────────────────────
    print("\n── Method Comparison ──────────────────────")
    print(f"{'Method':<12} {'PSNR (dB)':>10} "
          f"{'SSIM':>8} {'Bit Acc':>10}")
    print("-" * 44)
    for r in all_results:
        print(f"{r['method']:<12} "
              f"{r['avg_psnr_db']:>10.2f} "
              f"{r['avg_ssim']:>8.4f} "
              f"{r['avg_bit_accuracy']*100:>9.1f}%")
