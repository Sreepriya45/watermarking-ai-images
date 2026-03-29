# Watermarking AI-Generated Images
### Privacy and Provenance on Mobile Platforms
**CS8395 — Security & Privacy in Pervasive Computing**  
Sreepriya Damuluru & Jacqueline Frist | Vanderbilt University

---

## Project Overview
This project evaluates the robustness of watermarking 
methods for AI-generated images under real-world mobile 
transformation pipelines. We also investigate learning-based 
watermark removal attacks using a fine-tuned Vision 
Transformer, and analyze how watermarking affects fake 
image detection using SIDA 2025.

---

## Project Structure
```
watermarking-ai-images/
│
├── src/
│   ├── generate_dataset.py      # Generate AI images
│   ├── watermark_embed.py       # Embed watermarks
│   ├── mobile_transforms.py     # Mobile transformations
│   ├── vit_removal_attack.py    # ViT watermark removal
│   └── run_sida.py              # SIDA fake detection
│
├── dataset/
│   ├── original/                # AI generated images
│   ├── watermarked/             # Watermarked images
│   ├── transformed/             # After mobile transforms
│   └── removed/                 # After ViT removal
│
├── results/                     # Experiment results
├── models/                      # Saved model weights
├── requirements.txt
└── README.md
```

---

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate dataset
```bash
python src/generate_dataset.py
```

### 3. Embed watermarks
```bash
python src/watermark_embed.py
```

### 4. Run mobile transformations
```bash
python src/mobile_transforms.py
```

### 5. Train ViT removal attack
```bash
python src/vit_removal_attack.py
```

### 6. Run SIDA detection
```bash
python src/run_sida.py
```

---

## Results
Results are saved in the `results/` folder as JSON files.

---

## Requirements
- Python 3.9+
- PyTorch 2.0+
- CUDA GPU recommended
```

---

### Step 6 — Create .gitignore

Create a file called **.gitignore**:
```
# Python
__pycache__/
*.py[cod]
*.pyo
venv/
.env

# Dataset — too large for GitHub
dataset/
models/*.pth

# Results — commit manually
results/

# Mac
.DS_Store

# Jupyter
.ipynb_checkpoints/