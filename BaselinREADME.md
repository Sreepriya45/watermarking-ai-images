# Watermarking AI-Generated Images
### Privacy and Provenance on Mobile Platforms

## How to Run — Step by Step

### Step 1 — Generate AI Image Dataset
```bash
python3 src/generate_dataset.py
```

- Generates AI images using Stable Diffusion v1.5
- Saves to `data/original/`
- Baseline: 43 images across 5 prompts
- Uses Apple MPS (M4) for acceleration

### Step 2 — Embed Watermarks
```bash
python3 src/watermark_embed.py
```

- Embeds invisible watermarks using dwtDct method
- Measures PSNR and SSIM for image quality
- Saves watermarked images to `data/watermarked/`
- Results saved to `results/tables/embedding_results.json`

### Step 3 — Apply Mobile Transformations
```bash
python3 src/mobile_transforms.py
```

Applies 9 real-world mobile transformations:
- JPEG compression (50%, 70%, 90% quality)
- Cropping (25%, 50%)
- Resizing (50%, 75%)
- Gaussian blur
- Screenshot simulation

Results saved to `results/tables/transformation_results.json`

### Step 4 — ViT Watermark Removal Attack
```bash
python3 src/vit_removal_attack.py
python3 src/evaluate_removal.py
```

- Fine-tunes pre-trained ViT-B/16 on watermark pairs
- Trains on (watermarked image → clean image) pairs
- Evaluates bit accuracy before and after removal
- Results saved to `results/tables/vit_removal_results.json`

### Step 5 — Fake Image Detection
```bash
python3 src/run_sida.py
```

- Evaluates detection across 3 conditions:
  - Before watermarking
  - After watermarking
  - After ViT removal
- Results saved to `results/tables/sida_results.json`

---

## Baseline Results

### Watermark Embedding Quality

| Metric | Result | Target |
|--------|--------|--------|
| Avg PSNR | 39.89 dB | > 35 dB ✅ |
| Avg SSIM | 0.9707 | > 0.95 ✅ |
| Avg Bit Accuracy | 85.7% | > 80% ✅ |

### Mobile Transformation Robustness

| Transform | Bit Accuracy | Survival Rate |
|-----------|-------------|---------------|
| JPEG 50% | 49.7% | ❌ Destroyed |
| JPEG 70% | 49.9% | ❌ Destroyed |
| JPEG 90% | 54.0% | ❌ Barely survives |
| Crop 25% | 47.6% | ❌ Destroyed |
| Crop 50% | 54.7% | ❌ Barely survives |
| Resize 50% | 86.3% | ✅ Survives |
| Resize 75% | 86.4% | ✅ Survives |
| Blur | 62.2% | ⚠️ Partial |
| Screenshot | 85.7% | ✅ Survives |

### ViT Watermark Removal Attack

| Metric | Result |
|--------|--------|
| Bit Accuracy (watermarked) | 85.7% |
| Bit Accuracy (after removal) | 49.1% |
| Watermark Reduction | 36.6% |
| PSNR after removal | 12.80 dB |
| SSIM after removal | 0.2463 |

### Fake Image Detection (ResNet50 Proxy)

| Condition | Avg Score | Interpretation |
|-----------|-----------|----------------|
| Before Watermarking | 0.4322 | Baseline |
| After Watermarking | 0.4260 | Slightly harder to detect |
| After ViT Removal | 0.4685 | Easier to detect |

---

## Key Findings

**Finding 1 — JPEG compression destroys watermarks**
> JPEG compression at 50–70% quality reduces bit 
> accuracy to ~50% — essentially random, meaning the 
> watermark is completely destroyed. This is the most 
> common transformation applied by WhatsApp, Instagram, 
> and Telegram.

**Finding 2 — Resizing and screenshots preserve watermarks**
> Resizing and screenshot capture maintain bit accuracy 
> above 85%, suggesting the dwtDct watermark is robust 
> to spatial transformations but not frequency-based 
> compression.

**Finding 3 — ViT successfully removes watermarks**
> Fine-tuning ViT-B/16 on 43 image pairs reduces bit 
> accuracy from 85.7% to 49.1% — effectively destroying 
> the watermark. However, image quality degrades 
> (PSNR 12.80 dB) due to the small training dataset.

**Finding 4 — Watermark removal increases detectability**
> After ViT removal, fake detection score increases 
> from 0.4260 to 0.4685, suggesting the removal process 
> introduces visual artifacts that detectors can identify.

---

## Watermarking Methods

### DWT — Discrete Wavelet Transform
Decomposes the image into frequency sub-bands 
(low and high frequency components). Robust against 
resizing and cropping.

### DCT — Discrete Cosine Transform
Converts image to frequency domain using cosine 
functions. The same transform used in JPEG compression. 
Provides some resistance to compression attacks.

### SVD — Singular Value Decomposition
Decomposes image matrix into U, S, V components. 
Embedding in singular values provides stronger 
robustness against geometric attacks.

### Combined: dwtDct vs dwtDctSvd
- **dwtDct** (baseline): DWT + DCT — medium robustness
- **dwtDctSvd** (full run): DWT + DCT + SVD — higher 
  robustness, especially against compression

---

## Limitations (Baseline)

- Small dataset (43 images) — results may not generalize
- dwtDct method only — dwtDctSvd not yet evaluated
- ViT trained for only 10 epochs — underfitted
- ResNet50 used as proxy for SIDA (CVPR 2025)
- No real mobile platform testing (simulated only)

---

## Future Work (Full Run)

- Generate 200-500 images for more reliable results
- Compare dwtDct vs dwtDctSvd watermarking methods
- Train ViT for 30-50 epochs with larger dataset
- Integrate real SIDA model via Google Colab
- Test on actual WhatsApp/Instagram sharing pipeline

---

## Tools & Frameworks

| Tool | Purpose |
|------|---------|
| Stable Diffusion v1.5 | AI image generation |
| invisible-watermark | dwtDct/dwtDctSvd embedding |
| ViT-B/16 (timm) | Watermark removal attack |
| ResNet50 | Fake image detection proxy |
| PyTorch + MPS | M4 Mac acceleration |
| HuggingFace | Model loading |

---
