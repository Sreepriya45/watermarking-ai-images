"""
Build the 4 training pair types for ViT fine-tuning.

Pair 1 — real clean    → real clean     (identity, teaches model not to touch clean real images)
Pair 2 — real + wm     → real clean     (removal, teaches model to remove wm from real images)
Pair 3 — fake clean    → fake clean     (identity, teaches model not to touch clean AI images)
Pair 4 — fake + wm     → fake clean     (removal, teaches model to remove wm from AI images)

Usage:
    python3 src/build_four_pair_dataset.py

Outputs:
    data/real/                  real images (downloaded)
    data/real_watermarked/      real images with embedded watermark
    data/watermarked/           fake images with embedded watermark (already exists)
    data/pair_manifest.json     list of all (input, target, pair_type) triples
"""
import os
import json
import numpy as np
from PIL import Image
from tqdm import tqdm
from imwatermark import WatermarkEncoder

# ── Config ──────────────────────────────────────────────────────────────────
FAKE_DIR        = "data/original"       # existing AI-generated images
REAL_DIR        = "data/real"           # real images (downloaded here)
FAKE_WM_DIR     = "data/watermarked"    # fake + watermark (already built)
REAL_WM_DIR     = "data/real_watermarked"

WATERMARK_MSG   = "10101010110011001111000010101010"   # 32 bits, matches existing scripts
WATERMARK_METHOD = "dwtDct"

# How many real images to download. Keep proportional to your fake image count.
NUM_REAL_IMAGES  = 50

MANIFEST_PATH   = "data/pair_manifest.json"


# ── Watermark embedding ──────────────────────────────────────────────────────
def embed_watermark(img_path: str, out_path: str) -> None:
    """Embed invisible watermark into one image and save."""
    img = Image.open(img_path).convert("RGB")
    img_np = np.array(img)
    # invisible-watermark requires at least 256x256
    if img_np.shape[0] < 256 or img_np.shape[1] < 256:
        img = img.resize((256, 256), Image.LANCZOS)
        img_np = np.array(img)
    encoder = WatermarkEncoder()
    encoder.set_watermark("bits", WATERMARK_MSG)
    encoded = encoder.encode(img_np, WATERMARK_METHOD)
    Image.fromarray(encoded).save(out_path)


def embed_directory(src_dir: str, dst_dir: str, label: str) -> None:
    """Embed watermarks for every image in src_dir, save to dst_dir."""
    os.makedirs(dst_dir, exist_ok=True)
    files = [
        f for f in os.listdir(src_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    already_done = [f for f in files
                    if os.path.exists(
                        os.path.join(dst_dir, os.path.splitext(f)[0] + ".png")
                    )]
    remaining = [f for f in files if f not in already_done]

    if not remaining:
        print(f"  {label}: all {len(files)} images already watermarked. Skipping.")
        return

    print(f"\nEmbedding watermarks — {label} ({len(remaining)} images)...")
    for fname in tqdm(remaining, desc=label):
        src = os.path.join(src_dir, fname)
        base = os.path.splitext(fname)[0]
        dst = os.path.join(dst_dir, base + ".png")
        embed_watermark(src, dst)


# ── Step 1: Download real images from CelebA ────────────────────────────────
def download_real_images() -> None:
    """Download real face images from CelebA via HuggingFace datasets."""
    existing = [
        f for f in os.listdir(REAL_DIR)
        if f.endswith(".png")
    ] if os.path.exists(REAL_DIR) else []

    if len(existing) >= NUM_REAL_IMAGES:
        print(f"Real images already present ({len(existing)} found). Skipping download.")
        return

    os.makedirs(REAL_DIR, exist_ok=True)
    print(f"Downloading {NUM_REAL_IMAGES} real images from CelebA (HuggingFace)...")
    print("(First run may take a few minutes to stream the dataset.)\n")

    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' library not found.")
        print("Install it with:  pip install datasets")
        return

    # Try known working CelebA mirrors on HuggingFace in order.
    # Streams images one at a time so you never download the full dataset.
    DATASET_CANDIDATES = [
        "flwrlabs/celeba",
        "Andyrasika/CelebA",
        "tpremoli/CelebA-attrs",
    ]
    ds = None
    for candidate in DATASET_CANDIDATES:
        try:
            print(f"Trying dataset: {candidate} ...")
            ds = load_dataset(
                candidate,
                split="train",
                streaming=True,
                trust_remote_code=True,
            )
            # Peek at first example to confirm it loads
            first = next(iter(ds))
            img_check = first.get("image") or first.get("img") or first.get("Image")
            if img_check is None:
                raise ValueError("No image field found")
            print(f"Using dataset: {candidate}")
            # Re-create iterator from scratch (peek consumed one item)
            ds = load_dataset(
                candidate,
                split="train",
                streaming=True,
                trust_remote_code=True,
            )
            break
        except Exception as e:
            print(f"  Skipping {candidate}: {e}")
            ds = None

    if ds is None:
        print("\nERROR: Could not load any CelebA mirror from HuggingFace.")
        print("Try manually placing 50 face images (.png) into data/real/ and re-running.")
        return

    saved = len(existing)   # continue from where a previous run left off
    for example in tqdm(ds, desc="Downloading", total=NUM_REAL_IMAGES):
        if saved >= NUM_REAL_IMAGES:
            break
        img = example.get("image") or example.get("img") or example.get("Image")
        if img is None:
            continue
        if not isinstance(img, Image.Image):
            img = Image.fromarray(np.array(img))
        img = img.convert("RGB")
        out_path = os.path.join(REAL_DIR, f"real_{saved:04d}.png")
        img.save(out_path)
        saved += 1

    print(f"\n{saved} real images saved to {REAL_DIR}/")


# ── Step 2: Embed watermarks into real images ────────────────────────────────
def embed_real_watermarks() -> None:
    embed_directory(REAL_DIR, REAL_WM_DIR, "real images")


# ── Step 3: Confirm fake watermarks exist ───────────────────────────────────
def check_fake_watermarks() -> None:
    """
    data/watermarked/ should already exist from watermark_embed.py.
    If not, embed now.
    """
    if os.path.exists(FAKE_WM_DIR) and len(os.listdir(FAKE_WM_DIR)) > 0:
        count = len([f for f in os.listdir(FAKE_WM_DIR) if f.endswith(".png")])
        print(f"  Fake watermarked images: {count} found in {FAKE_WM_DIR}/. OK.")
    else:
        print(f"  {FAKE_WM_DIR}/ missing — embedding now...")
        embed_directory(FAKE_DIR, FAKE_WM_DIR, "fake images")


# ── Step 4: Build manifest ───────────────────────────────────────────────────
def build_manifest() -> None:
    """
    Write a JSON file listing every (input_path, target_path, pair_type, label).
    The ViT training script reads this instead of scanning directories directly.
    """
    pairs = []

    def add_pairs(input_dir, target_dir, pair_type, label):
        if not os.path.exists(input_dir):
            print(f"  WARNING: {input_dir} not found — skipping {label}")
            return
        if not os.path.exists(target_dir):
            print(f"  WARNING: {target_dir} not found — skipping {label}")
            return

        files = sorted([
            f for f in os.listdir(input_dir) if f.endswith(".png")
        ])
        added = 0
        for fname in files:
            inp = os.path.join(input_dir, fname)
            tgt = os.path.join(target_dir, fname)
            if not os.path.exists(tgt):
                continue
            pairs.append({
                "input":     inp,
                "target":    tgt,
                "pair_type": pair_type,
                "label":     label,
            })
            added += 1
        print(f"  Pair {pair_type} ({label}): {added} pairs")

    print("\nBuilding pair manifest...")

    # Pair 1: real clean  →  real clean  (identity)
    add_pairs(REAL_DIR,    REAL_DIR,    1, "real_clean_identity")

    # Pair 2: real + wm  →  real clean  (removal)
    add_pairs(REAL_WM_DIR, REAL_DIR,    2, "real_watermark_removal")

    # Pair 3: fake clean  →  fake clean  (identity)
    add_pairs(FAKE_DIR,    FAKE_DIR,    3, "fake_clean_identity")

    # Pair 4: fake + wm  →  fake clean  (removal)
    add_pairs(FAKE_WM_DIR, FAKE_DIR,    4, "fake_watermark_removal")

    os.makedirs(os.path.dirname(MANIFEST_PATH) or ".", exist_ok=True)
    with open(MANIFEST_PATH, "w") as f:
        json.dump(pairs, f, indent=2)

    print(f"\n── Manifest Summary ────────────────────────")
    by_label = {}
    for p in pairs:
        by_label.setdefault(p["label"], 0)
        by_label[p["label"]] += 1
    for label, count in by_label.items():
        print(f"  {label:<30}: {count:>4} pairs")
    print(f"  {'TOTAL':<30}: {len(pairs):>4} pairs")
    print(f"\nManifest saved → {MANIFEST_PATH}")


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("Building 4-pair dataset for ViT fine-tuning")
    print("=" * 55)

    print("\n[1/4] Real image download")
    download_real_images()

    print("\n[2/4] Embed watermarks into real images")
    embed_real_watermarks()

    print("\n[3/4] Check fake watermarked images")
    check_fake_watermarks()

    print("\n[4/4] Build pair manifest")
    build_manifest()

    print("\nDone! Next step:")
    print("  python3 src/vit_removal_attack.py")
