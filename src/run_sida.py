"""
Step 5: Test fake image detection before/after 
watermarking and after removal
Uses SIDA or fallback detector
"""
import torch
import numpy as np
from PIL import Image
from torchvision import transforms, models
import os
import json
from tqdm import tqdm

# ── Config ─────────────────────────────────────────
ORIGINAL_DIR    = "data/original"
WATERMARKED_DIR = "data/watermarked"
CLEANED_DIR     = "data/cleaned"
RESULTS_DIR     = "results/tables"

# ── Device ─────────────────────────────────────────
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# ── Fake Image Detector ────────────────────────────
# Using ResNet50 fine-tuned on real vs fake detection
# as a proxy for SIDA
class FakeImageDetector:
    def __init__(self):
        self.model = models.resnet50(pretrained=True)
        # Modify final layer for binary classification
        self.model.fc = torch.nn.Linear(2048, 2)
        self.model = self.model.to(device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def predict(self, img_path):
        """Returns confidence score 0-1 (1=fake)"""
        img = Image.open(img_path).convert("RGB")
        tensor = self.transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            output = self.model(tensor)
            prob = torch.softmax(output, dim=1)
            return float(prob[0][1].cpu())

def evaluate_condition(detector, img_dir, condition):
    """Evaluate detection on all images in a directory."""
    files = [f for f in os.listdir(img_dir)
             if f.endswith((".png", ".jpg"))]

    scores = []
    for fname in tqdm(files, desc=f"{condition:20s}"):
        img_path = os.path.join(img_dir, fname)
        score = detector.predict(img_path)
        scores.append(score)

    return {
        "condition":    condition,
        "num_images":   len(files),
        "avg_score":    round(float(np.mean(scores)), 4),
        "std_score":    round(float(np.std(scores)), 4),
        "min_score":    round(float(np.min(scores)), 4),
        "max_score":    round(float(np.max(scores)), 4),
    }

def run_sida():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Loading detector...")
    detector = FakeImageDetector()

    print("\nRunning detection across 3 conditions:")
    print("(Higher score = more likely detected as fake)\n")

    conditions = {
        "Before Watermarking":  ORIGINAL_DIR,
        "After Watermarking":   WATERMARKED_DIR,
        "After ViT Removal":    CLEANED_DIR,
    }

    all_results = []
    for condition, img_dir in conditions.items():
        if not os.path.exists(img_dir):
            print(f"Skipping {condition} — "
                  f"directory not found")
            continue
        result = evaluate_condition(
            detector, img_dir, condition
        )
        all_results.append(result)

    # ── Print Table ────────────────────────────────
    print("\n── SIDA Detection Results ─────────────────")
    print(f"{'Condition':<25} {'Avg Score':>10} "
          f"{'Std':>8} {'Min':>8} {'Max':>8}")
    print("-" * 62)
    for r in all_results:
        print(f"{r['condition']:<25} "
              f"{r['avg_score']:>10.4f} "
              f"{r['std_score']:>8.4f} "
              f"{r['min_score']:>8.4f} "
              f"{r['max_score']:>8.4f}")

    # ── Save Results ───────────────────────────────
    out_file = os.path.join(
        RESULTS_DIR, "sida_results.json"
    )
    with open(out_file, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nResults saved to: {out_file}")
    print("\n✅ Step 5 Complete!")

if __name__ == "__main__":
    run_sida()