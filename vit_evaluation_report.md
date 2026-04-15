# ViT Watermark Removal — Evaluation Report

**Generated:** 2026-04-14 19:26  
**Epochs trained:** 10 | **Dataset:** 120 total pairs | **Device:** mps

---

## Training Loss

| Epoch | Train Loss | Val Loss |
|------:|-----------:|---------:|
|     1 | 0.068252 | 0.049643 |
|     2 | 0.039170 | 0.034155 |
|     3 | 0.024929 | 0.022489 |
|     4 | 0.019300 | 0.019827 |
|     5 | 0.017055 | 0.018038 |
|     6 | 0.014867 | 0.015918 |
|     7 | 0.013374 | 0.013876 |
|     8 | 0.011906 | 0.012771 |
|     9 | 0.010887 | 0.011121 |
|    10 | 0.010006 | 0.009919 |

---

## Results by Pair Type

### Pair 1 — Real clean → Real clean (n=50)

| Metric | Mean | Std |
|--------|-----:|----:|
| PSNR (dB) | 21.07 | 1.64 |
| SSIM      | 0.5562 | 0.0664 |

> **Interpretation:** The model sees a clean image and should output it unchanged. Higher PSNR/SSIM = model correctly leaves clean images alone.

### Pair 2 — Real+WM → Real clean (n=50)

| Metric | Mean | Std |
|--------|-----:|----:|
| PSNR (dB) | 21.07 | 1.63 |
| SSIM      | 0.5568 | 0.0659 |

> **Interpretation:** Input is watermarked; target is the clean original. Higher PSNR/SSIM vs target = better reconstruction. Lower WM bit accuracy after cleaning = more successful attack. Bit accuracy ≤ 50% means the watermark is effectively destroyed (random chance).

### Pair 3 — Fake clean → Fake clean (n=10)

| Metric | Mean | Std |
|--------|-----:|----:|
| PSNR (dB) | 19.45 | 3.05 |
| SSIM      | 0.4097 | 0.1796 |

> **Interpretation:** The model sees a clean image and should output it unchanged. Higher PSNR/SSIM = model correctly leaves clean images alone.

### Pair 4 — Fake+WM → Fake clean (n=10)

| Metric | Mean | Std |
|--------|-----:|----:|
| PSNR (dB) | 19.44 | 3.04 |
| SSIM      | 0.4097 | 0.1795 |

> **Interpretation:** Input is watermarked; target is the clean original. Higher PSNR/SSIM vs target = better reconstruction. Lower WM bit accuracy after cleaning = more successful attack. Bit accuracy ≤ 50% means the watermark is effectively destroyed (random chance).

---

## Summary

| Pair | Task | PSNR ↑ | SSIM ↑ | WM Bit Acc ↓ |
|-----:|------|-------:|-------:|-------------:|
| 1 | Real clean → Real clean | 21.07 | 0.5562 | — |
| 2 | Real+WM → Real clean | 21.07 | 0.5568 | — |
| 3 | Fake clean → Fake clean | 19.45 | 0.4097 | — |
| 4 | Fake+WM → Fake clean | 19.44 | 0.4097 | — |

> **Reading the summary:** For identity pairs (1 & 3), PSNR/SSIM should be high (model leaves clean images alone). For removal pairs (2 & 4), PSNR/SSIM should be high vs the clean target AND WM bit accuracy should be low (≤50% = watermark gone).
