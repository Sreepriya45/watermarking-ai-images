"""
Step 4: Fine-tune pre-trained ViT to remove watermarks

Now uses a 4-pair dataset manifest (built by build_four_pair_dataset.py):
  Pair 1 — real clean    → real clean     (identity)
  Pair 2 — real + wm     → real clean     (removal)
  Pair 3 — fake clean    → fake clean     (identity)
  Pair 4 — fake + wm     → fake clean     (removal)
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import timm
import os
import numpy as np
from tqdm import tqdm
import json
import datetime

# ── Config ─────────────────────────────────────────
MANIFEST_PATH    = "data/pair_manifest.json"   # built by build_four_pair_dataset.py
CLEANED_DIR      = "data/cleaned"
MODELS_DIR       = "models"
RESULTS_DIR      = "results/tables"
REPORT_PATH      = "vit_evaluation_report.md"
IMG_SIZE         = 224
BATCH_SIZE       = 4
NUM_EPOCHS       = 10
LEARNING_RATE    = 1e-4

# Must match the values used in build_four_pair_dataset.py
WATERMARK_MSG    = "10101010110011001111000010101010"
WATERMARK_METHOD = "dwtDct"

# ── Device ─────────────────────────────────────────
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")   # Apple M4
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# ── Dataset ────────────────────────────────────────
class FourPairDataset(Dataset):
    """
    Reads all training pairs from the manifest produced by
    build_four_pair_dataset.py. Each entry has:
        input  — path to the image fed to the model
        target — path to the image the model should output
        pair_type — 1..4 (see module docstring)
    """
    def __init__(self, manifest_path):
        if not os.path.exists(manifest_path):
            raise FileNotFoundError(
                f"Manifest not found: {manifest_path}\n"
                "Run:  python3 src/build_four_pair_dataset.py"
            )
        with open(manifest_path) as f:
            self.pairs = json.load(f)

        self.transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
        ])

        by_type = {}
        for p in self.pairs:
            by_type.setdefault(p["label"], 0)
            by_type[p["label"]] += 1
        print(f"Dataset: {len(self.pairs)} total pairs")
        for label, count in by_type.items():
            print(f"  {label}: {count}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        entry = self.pairs[idx]
        inp = Image.open(entry["input"]).convert("RGB")
        tgt = Image.open(entry["target"]).convert("RGB")
        fname = os.path.basename(entry["input"])
        return (
            self.transform(inp),
            self.transform(tgt),
            fname,
        )

# ── Model — Fine-tuned ViT ─────────────────────────
class ViTWatermarkRemover(nn.Module):
    """
    ViT-B/16 encoder + spatial convolutional decoder.

    The encoder's forward_features() returns all 197 tokens:
      token 0   — CLS (global summary, discarded here)
      tokens 1..196 — one 768-dim vector per 16x16 patch → 14x14 spatial grid

    The decoder upsamples that 14x14x768 feature map back to 224x224x3
    using transposed convolutions, preserving per-image spatial detail.
    Without this, the model collapses to outputting the same average image
    for every input (information bottleneck from a single 768-dim vector).
    """
    def __init__(self):
        super().__init__()
        # Load pre-trained ViT-B/16; num_classes=0 removes the classifier head
        # but we call forward_features() directly so we get all patch tokens.
        self.encoder = timm.create_model(
            'vit_base_patch16_224',
            pretrained=True,
            num_classes=0
        )
        # Freeze early layers — only fine-tune last 4 transformer blocks
        blocks = list(self.encoder.blocks)
        for block in blocks[:-4]:
            for param in block.parameters():
                param.requires_grad = False

        # Spatial decoder: (B, 768, 14, 14) → (B, 3, 224, 224)
        # Each ConvTranspose2d doubles the spatial resolution.
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(768, 256, kernel_size=4, stride=2, padding=1),  # → 28x28
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # → 56x56
            nn.ReLU(),
            nn.ConvTranspose2d(128,  64, kernel_size=4, stride=2, padding=1),  # → 112x112
            nn.ReLU(),
            nn.ConvTranspose2d( 64,   3, kernel_size=4, stride=2, padding=1),  # → 224x224
            nn.Sigmoid()
        )

    def forward(self, x):
        # Returns (B, 197, 768): CLS token + 196 patch tokens
        all_tokens = self.encoder.forward_features(x)
        # Drop CLS, keep patch tokens → (B, 196, 768)
        patch_tokens = all_tokens[:, 1:, :]
        # Reshape to spatial grid: 196 patches = 14x14
        B = patch_tokens.shape[0]
        spatial = patch_tokens.transpose(1, 2).reshape(B, 768, 14, 14)
        # Decode back to full image
        return self.decoder(spatial)   # (B, 3, 224, 224)

# ── Evaluation helpers ─────────────────────────────
PAIR_META = {
    "real_clean_identity":    (1, "Real clean → Real clean",   "identity"),
    "real_watermark_removal": (2, "Real+WM → Real clean",      "removal"),
    "fake_clean_identity":    (3, "Fake clean → Fake clean",   "identity"),
    "fake_watermark_removal": (4, "Fake+WM → Fake clean",      "removal"),
}

INTERPRETATIONS = {
    "identity": (
        "The model sees a clean image and should output it unchanged. "
        "Higher PSNR/SSIM = model correctly leaves clean images alone."
    ),
    "removal": (
        "Input is watermarked; target is the clean original. "
        "Higher PSNR/SSIM vs target = better reconstruction. "
        "Lower WM bit accuracy after cleaning = more successful attack. "
        "Bit accuracy ≤ 50% means the watermark is effectively destroyed (random chance)."
    ),
}


def _write_report(by_label, history, n_total, report_path):
    """Write evaluation results to a Markdown file."""
    os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    lines = []
    lines += [
        "# ViT Watermark Removal — Evaluation Report\n",
        f"**Generated:** {ts}  ",
        f"**Epochs trained:** {NUM_EPOCHS} | "
        f"**Dataset:** {n_total} total pairs | "
        f"**Device:** {device}\n",
        "---\n",
        "## Training Loss\n",
        "| Epoch | Train Loss | Val Loss |",
        "|------:|-----------:|---------:|",
    ]
    for h in history:
        lines.append(
            f"| {h['epoch']:>5} | {h['train_loss']:.6f} | {h['val_loss']:.6f} |"
        )

    lines += ["", "---\n", "## Results by Pair Type\n"]

    summary_rows = []
    for label, entries in sorted(by_label.items(), key=lambda kv: PAIR_META[kv[0]][0]):
        pair_num, title, task = PAIR_META[label]
        n = len(entries)
        psnrs = [e["psnr"] for e in entries]
        ssims = [e["ssim"] for e in entries]

        lines += [
            f"### Pair {pair_num} — {title} (n={n})\n",
            "| Metric | Mean | Std |",
            "|--------|-----:|----:|",
            f"| PSNR (dB) | {np.mean(psnrs):.2f} | {np.std(psnrs):.2f} |",
            f"| SSIM      | {np.mean(ssims):.4f} | {np.std(ssims):.4f} |",
        ]

        wm_before = [e["wm_acc_before"] for e in entries if "wm_acc_before" in e]
        wm_after  = [e["wm_acc_after"]  for e in entries if "wm_acc_after"  in e]
        wm_col = "—"
        if wm_before and wm_after:
            mean_before = np.mean(wm_before) * 100
            mean_after  = np.mean(wm_after)  * 100
            det_rate    = np.mean([a > 0.75 for a in wm_after]) * 100
            lines += [
                f"| WM bit accuracy — input   | {mean_before:.1f}% | — |",
                f"| WM bit accuracy — cleaned | {mean_after:.1f}% | — |",
                f"| WM detection rate — cleaned (>75% threshold) | {det_rate:.1f}% | — |",
            ]
            wm_col = f"{mean_after:.1f}%"

        lines += [
            "",
            f"> **Interpretation:** {INTERPRETATIONS[task]}\n",
        ]
        summary_rows.append((
            pair_num, title,
            f"{np.mean(psnrs):.2f}", f"{np.mean(ssims):.4f}", wm_col,
        ))

    lines += [
        "---\n",
        "## Summary\n",
        "| Pair | Task | PSNR ↑ | SSIM ↑ | WM Bit Acc ↓ |",
        "|-----:|------|-------:|-------:|-------------:|",
    ]
    for row in summary_rows:
        lines.append(f"| {row[0]} | {row[1]} | {row[2]} | {row[3]} | {row[4]} |")

    lines += [
        "",
        "> **Reading the summary:** For identity pairs (1 & 3), PSNR/SSIM should be high "
        "(model leaves clean images alone). For removal pairs (2 & 4), PSNR/SSIM should be "
        "high vs the clean target AND WM bit accuracy should be low (≤50% = watermark gone).",
    ]

    with open(report_path, "w") as f:
        f.write("\n".join(lines) + "\n")


def evaluate_and_report(model, dataset, history):
    """
    Run the trained model over every image in the dataset, save cleaned outputs,
    compute PSNR/SSIM vs the clean target, and for watermarked-input pairs (2 & 4)
    also measure how much of the watermark survives in the cleaned image.
    Writes a summary Markdown report to REPORT_PATH.
    """
    try:
        from skimage.metrics import peak_signal_noise_ratio, structural_similarity
    except ImportError:
        print("WARNING: scikit-image not found — install with: pip install scikit-image")
        peak_signal_noise_ratio = structural_similarity = None

    try:
        from imwatermark import WatermarkDecoder
        wm_decoder = WatermarkDecoder("bits", len(WATERMARK_MSG))
    except Exception:
        wm_decoder = None

    os.makedirs(CLEANED_DIR, exist_ok=True)
    by_label = {}

    model.eval()
    print("\nEvaluating and saving cleaned images...")
    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc="Evaluating"):
            pair_info       = dataset.pairs[idx]
            inp_t, tgt_t, fname = dataset[idx]

            # Run model
            out_t  = model(inp_t.unsqueeze(0).to(device))[0].cpu()
            out_np = out_t.numpy().transpose(1, 2, 0)          # HWC float [0,1]
            tgt_np = tgt_t.numpy().transpose(1, 2, 0)

            # Save cleaned image
            out_uint8 = (out_np * 255).astype(np.uint8)
            Image.fromarray(out_uint8).save(os.path.join(CLEANED_DIR, fname))

            entry = {"pair_type": pair_info["pair_type"]}

            # PSNR & SSIM vs clean target
            if peak_signal_noise_ratio is not None:
                entry["psnr"] = peak_signal_noise_ratio(tgt_np, out_np, data_range=1.0)
                entry["ssim"] = structural_similarity(
                    tgt_np, out_np, channel_axis=-1, data_range=1.0
                )
            else:
                entry["psnr"] = float("nan")
                entry["ssim"] = float("nan")

            # Watermark detection for removal pairs (types 2 & 4)
            if pair_info["pair_type"] in (2, 4) and wm_decoder is not None:
                inp_np    = inp_t.numpy().transpose(1, 2, 0)
                inp_uint8 = (inp_np * 255).astype(np.uint8)
                # imwatermark uses OpenCV (BGR) internally
                inp_bgr = inp_uint8[:, :, ::-1]
                out_bgr = out_uint8[:, :, ::-1]
                try:
                    bits_before = wm_decoder.decode(inp_bgr, WATERMARK_METHOD)
                    bits_after  = wm_decoder.decode(out_bgr, WATERMARK_METHOD)
                    def bit_acc(bits):
                        return sum(a == b for a, b in zip(bits, WATERMARK_MSG)) / len(WATERMARK_MSG)
                    entry["wm_acc_before"] = bit_acc(bits_before)
                    entry["wm_acc_after"]  = bit_acc(bits_after)
                except Exception:
                    pass

            by_label.setdefault(pair_info["label"], []).append(entry)

    print(f"Cleaned images saved → {CLEANED_DIR}/")
    _write_report(by_label, history, len(dataset), REPORT_PATH)
    print(f"Evaluation report saved → {REPORT_PATH}")


# ── Training ───────────────────────────────────────
def train():
    os.makedirs(CLEANED_DIR,  exist_ok=True)
    os.makedirs(MODELS_DIR,   exist_ok=True)
    os.makedirs(RESULTS_DIR,  exist_ok=True)

    dataset = FourPairDataset(MANIFEST_PATH)

    if len(dataset) == 0:
        print("ERROR: No matching pairs found!")
        return

    # Split 80/20
    train_size = max(1, int(0.8 * len(dataset)))
    test_size  = len(dataset) - train_size
    train_set, test_set = torch.utils.data.random_split(
        dataset, [train_size, test_size]
    )

    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True
    )
    test_loader = DataLoader(
        test_set, batch_size=BATCH_SIZE
    )

    print(f"\nTrain: {len(train_set)} | "
          f"Test: {len(test_set)}")

    model     = ViTWatermarkRemover().to(device)
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad,
               model.parameters()),
        lr=LEARNING_RATE
    )
    criterion = nn.MSELoss()

    history = []

    print(f"\nFine-tuning ViT for {NUM_EPOCHS} epochs...")

    for epoch in range(NUM_EPOCHS):
        # ── Train ──────────────────────────────────
        model.train()
        train_loss = 0.0

        for watermarked, clean, _ in tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"
        ):
            watermarked = watermarked.to(device)
            clean       = clean.to(device)

            optimizer.zero_grad()
            output = model(watermarked)
            loss   = criterion(output, clean)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train = train_loss / len(train_loader)

        # ── Validate ───────────────────────────────
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for watermarked, clean, _ in test_loader:
                watermarked = watermarked.to(device)
                clean       = clean.to(device)
                output      = model(watermarked)
                loss        = criterion(output, clean)
                val_loss   += loss.item()

        avg_val = val_loss / max(1, len(test_loader))

        print(f"  Epoch {epoch+1:2d} — "
              f"Train Loss: {avg_train:.6f}  "
              f"Val Loss: {avg_val:.6f}")

        history.append({
            "epoch":      epoch + 1,
            "train_loss": round(avg_train, 6),
            "val_loss":   round(avg_val, 6),
        })

    # ── Save Model ─────────────────────────────────
    model_path = os.path.join(
        MODELS_DIR, "vit_remover.pth"
    )
    torch.save(model.state_dict(), model_path)
    print(f"\nModel saved to: {model_path}")

    # ── Evaluate + Save Cleaned Images + Write Report ─
    evaluate_and_report(model, dataset, history)
    print("\n✅ Step 4 Complete!")

if __name__ == "__main__":
    train()