"""
Step 3: Apply mobile transformations and measure
watermark survival rate — comparing dwtDct vs dwtDctSvd
"""
import numpy as np
from PIL import Image
from imwatermark import WatermarkDecoder
import os
import json
import cv2
from tqdm import tqdm

# ── Config ─────────────────────────────────────────
RESULTS_DIR   = "results/tables"
WATERMARK_MSG = "10101010110011001111000010101010"
WATERMARK_BITS = 32

WATERMARKED_DIRS = {
    "dwtDct":    "data/watermarked",
    "dwtDctSvd": "data/watermarked_dwtDctSvd",
}
ATTACKED_DIRS = {
    "dwtDct":    "data/attacked_dwtDct",
    "dwtDctSvd": "data/attacked_dwtDctSvd",
}

# ── Detection ──────────────────────────────────────
def detect_watermark(img_path, method):
    img_np = np.array(
        Image.open(img_path).convert("RGB")
    )
    decoder = WatermarkDecoder('bits', WATERMARK_BITS)
    return decoder.decode(img_np, method)

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
    return fname

def run_transforms_for_method(method):
    """Run all transformations for one watermarking method."""
    watermarked_dir = WATERMARKED_DIRS[method]
    attacked_dir    = ATTACKED_DIRS[method]

    if not os.path.exists(watermarked_dir):
        print(f"  Skipping {method} — "
              f"{watermarked_dir} not found")
        return None

    files = [f for f in os.listdir(watermarked_dir)
             if f.endswith((".png", ".jpg"))]

    print(f"\n[{method}] {len(files)} images, "
          f"{len(TRANSFORMS)} transformations")

    summary = {}

    for t_name, t_fn in TRANSFORMS.items():
        out_dir = os.path.join(attacked_dir, t_name)
        os.makedirs(out_dir, exist_ok=True)

        acc_list = []

        for fname in tqdm(files, desc=f"{t_name:12s}"):
            in_path  = os.path.join(watermarked_dir, fname)
            out_fname = get_out_fname(fname, t_name)
            out_path  = os.path.join(out_dir, out_fname)

            try:
                t_fn(in_path, out_path)
                detected = detect_watermark(out_path, method)
                acc = bit_accuracy(detected)
            except Exception as e:
                print(f"  Error on {fname}: {e}")
                acc = 0.0

            acc_list.append(acc)

        avg_acc = float(np.mean(acc_list))
        summary[t_name] = {
            "avg_bit_accuracy":  round(avg_acc, 4),
            "survival_rate_pct": round(avg_acc * 100, 1),
        }

    out_file = os.path.join(
        RESULTS_DIR, f"transformation_{method}.json"
    )
    with open(out_file, "w") as f:
        json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)

    all_results = {}
    for method in ["dwtDct", "dwtDctSvd"]:
        result = run_transforms_for_method(method)
        if result:
            all_results[method] = result

    if len(all_results) < 2:
        print("\nOnly one method ran — skipping comparison.")
    else:
        # ── Print Side-by-Side Table ───────────────
        print("\n── Robustness Comparison ──────────────────")
        print(f"{'Transform':<15} {'dwtDct':>10} "
              f"{'dwtDctSvd':>12}")
        print("-" * 40)
        for t in TRANSFORMS:
            d1 = all_results["dwtDct"][t]["survival_rate_pct"]
            d2 = all_results["dwtDctSvd"][t]["survival_rate_pct"]
            print(f"{t:<15} {d1:>9.1f}% {d2:>11.1f}%")
