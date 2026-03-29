"""
Step 4: Fine-tune pre-trained ViT to remove watermarks
As suggested by Professor Linda
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

# ── Config ─────────────────────────────────────────
CLEAN_DIR       = "data/original"
WATERMARKED_DIR = "data/watermarked"
CLEANED_DIR     = "data/cleaned"
MODELS_DIR      = "models"
RESULTS_DIR     = "results/tables"
IMG_SIZE        = 224
BATCH_SIZE      = 4
NUM_EPOCHS      = 10
LEARNING_RATE   = 1e-4

# ── Device ─────────────────────────────────────────
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")   # Apple M4
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# ── Dataset ────────────────────────────────────────
class WatermarkPairDataset(Dataset):
    def __init__(self, clean_dir, watermarked_dir):
        self.clean_dir      = clean_dir
        self.watermarked_dir = watermarked_dir
        self.files = [
            f for f in os.listdir(watermarked_dir)
            if f.endswith(".png")
            and os.path.exists(
                os.path.join(clean_dir, f)
            )
        ]
        self.transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
        ])
        print(f"Dataset: {len(self.files)} pairs found")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        clean = Image.open(
            os.path.join(self.clean_dir, fname)
        ).convert("RGB")
        watermarked = Image.open(
            os.path.join(self.watermarked_dir, fname)
        ).convert("RGB")
        return (
            self.transform(watermarked),
            self.transform(clean),
            fname
        )

# ── Model — Fine-tuned ViT ─────────────────────────
class ViTWatermarkRemover(nn.Module):
    def __init__(self):
        super().__init__()
        # Load pre-trained ViT-B/16
        # (as suggested by Professor Linda)
        self.encoder = timm.create_model(
            'vit_base_patch16_224',
            pretrained=True,
            num_classes=0  # remove classifier
        )
        # Freeze early layers — only fine-tune last 4
        blocks = list(self.encoder.blocks)
        for block in blocks[:-4]:
            for param in block.parameters():
                param.requires_grad = False

        # Decoder to reconstruct clean image
        self.decoder = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, IMG_SIZE * IMG_SIZE * 3),
            nn.Sigmoid()
        )

    def forward(self, x):
        features = self.encoder(x)       # (B, 768)
        output   = self.decoder(features) # (B, H*W*3)
        output   = output.view(
            -1, 3, IMG_SIZE, IMG_SIZE
        )
        return output

# ── Training ───────────────────────────────────────
def train():
    os.makedirs(CLEANED_DIR,  exist_ok=True)
    os.makedirs(MODELS_DIR,   exist_ok=True)
    os.makedirs(RESULTS_DIR,  exist_ok=True)

    dataset = WatermarkPairDataset(
        CLEAN_DIR, WATERMARKED_DIR
    )

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
    print("(This is Professor Linda's suggestion)\n")

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

    # ── Save Cleaned Images ────────────────────────
    print("\nGenerating cleaned images...")
    model.eval()
    with torch.no_grad():
        for watermarked, clean, fnames in DataLoader(
            dataset, batch_size=1
        ):
            watermarked = watermarked.to(device)
            output = model(watermarked)
            for i, fname in enumerate(fnames):
                img_np = (
                    output[i].cpu().numpy()
                    .transpose(1, 2, 0) * 255
                ).astype(np.uint8)
                Image.fromarray(img_np).save(
                    os.path.join(CLEANED_DIR, fname)
                )

    print(f"Cleaned images saved to: {CLEANED_DIR}/")

    # ── Save Training History ──────────────────────
    history_file = os.path.join(
        RESULTS_DIR, "vit_training_history.json"
    )
    with open(history_file, "w") as f:
        json.dump(history, f, indent=2)

    print(f"Training history saved to: {history_file}")
    print("\n✅ Step 4 Complete!")

if __name__ == "__main__":
    train()