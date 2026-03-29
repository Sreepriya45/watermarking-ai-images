"""
Step 3: Apply mobile transformations and measure
watermark survival rate
"""
import numpy as np
from PIL import Image
from imwatermark import WatermarkDecoder
import os
import json
import cv2
from tqdm import tqdm

# ── Config ─────────────────────────────────────────
WATERMARKED_DIR = "data/watermarked"
ATTACKED_DIR    = "data/attacked"
RESULTS_DIR     = "results/tables"
WATERMARK_MSG   = "10101010110011001111000010101010"
WATERMARK_BITS  = 32
METHOD          = "dwtDct"

# ── Detection ──────────────────────────────────────
def detect_watermark(img_path):
    img_np = np.array(
        Image.open(img_path).convert("RGB")
    )
    decoder = WatermarkDecoder('bits', WATERMARK_BITS)
    return decoder.decode(img_np, METHOD)

def bit_accuracy(detected, original=WATERMARK_MSG):
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

# ── Transformations ────────────────────────────────
def jpeg_compress(img_path, out_path, quality):
    img = Image.open(img_path).convert("RGB")
    img.save(out_path, "JPEG", quality=quality)

def crop_image(img_path, out_path, percent):
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    px_w = int(w * percent / 2)
    px_h = int(h * percent / 2)
    cropped = img.crop((px_w, px_h,
                        w - px_w, h - px_h))
    cropped = cropped.resize((w, h), Image.LANCZOS)
    cropped.save(out_path)

def resize_image(img_path, out_path, scale):
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    small = img.resize(
        (int(w * scale), int(h * scale)), Image.LANCZOS
    )
    restored = small.resize((w, h), Image.LANCZOS)
    restored.save(out_path)

def blur_image(img_path, out_path, kernel=5):
    img = cv2.imread(img_path)
    blurred = cv2.GaussianBlur(img, (kernel, kernel), 0)
    cv2.imwrite(out_path, blurred)

def screenshot_sim(img_path, out_path):
    """Simulate screenshot — always saves as PNG."""
    img = Image.open(img_path).convert("RGB")
    img.save(out_path, "PNG")
    img2 = Image.open(out_path).convert("RGB")
    img2.save(out_path, "PNG", compress_level=6)

# ── All Transformations ────────────────────────────
TRANSFORMS = {
    "jpeg_50":    lambda i, o: jpeg_compress(i, o, 50),
    "jpeg_70":    lambda i, o: jpeg_compress(i, o, 70),
    "jpeg_90":    lambda i, o: jpeg_compress(i, o, 90),
    "crop_25":    lambda i, o: crop_image(i, o, 0.25),
    "crop_50":    lambda i, o: crop_image(i, o, 0.50),
    "resize_50":  lambda i, o: resize_image(i, o, 0.5),
    "resize_75":  lambda i, o: resize_image(i, o, 0.75),
    "blur":       lambda i, o: blur_image(i, o),
    "screenshot": lambda i, o: screenshot_sim(i, o),
}

def get_out_fname(fname, t_name):
    """Get output filename — jpeg gets .jpg, rest stay .png"""
    if "jpeg" in t_name:
        return fname.replace(".png", ".jpg")
    return fname  # always .png for everything else

def run_all_transforms():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    files = [f for f in os.listdir(WATERMARKED_DIR)
             if f.endswith((".png", ".jpg"))]

    print(f"Found {len(files)} watermarked images")
    print(f"Running {len(TRANSFORMS)} transformations\n")

    summary = {}

    for t_name, t_fn in TRANSFORMS.items():
        out_dir = os.path.join(ATTACKED_DIR, t_name)
        os.makedirs(out_dir, exist_ok=True)

        acc_list = []

        for fname in tqdm(files, desc=f"{t_name:12s}"):
            in_path  = os.path.join(WATERMARKED_DIR, fname)
            out_fname = get_out_fname(fname, t_name)
            out_path  = os.path.join(out_dir, out_fname)

            try:
                t_fn(in_path, out_path)
                detected = detect_watermark(out_path)
                acc = bit_accuracy(detected)
            except Exception as e:
                print(f"  Error on {fname}: {e}")
                acc = 0.0

            acc_list.append(acc)

        avg_acc = float(np.mean(acc_list))
        summary[t_name] = {
            "avg_bit_accuracy":  round(avg_acc, 4),
            "survival_rate_pct": round(avg_acc * 100, 1)
        }
        print(f"  {t_name:12s} → "
              f"Bit Accuracy: {avg_acc * 100:.1f}%")

    # ── Save Results ───────────────────────────────
    out_file = os.path.join(
        RESULTS_DIR, "transformation_results.json"
    )
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)

    # ── Print Final Table ──────────────────────────
    print("\n── Transformation Results ─────────────────")
    print(f"{'Transform':<15} {'Bit Accuracy':>15} "
          f"{'Survival Rate':>15}")
    print("-" * 47)
    for t_name, res in summary.items():
        print(f"{t_name:<15} "
              f"{res['avg_bit_accuracy']:>15.4f} "
              f"{res['survival_rate_pct']:>14.1f}%")

    print(f"\nResults saved to: {out_file}")
    print(f"Attacked images saved to: {ATTACKED_DIR}/")

if __name__ == "__main__":
    run_all_transforms()