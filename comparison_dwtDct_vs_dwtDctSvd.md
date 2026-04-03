# dwtDct vs dwtDctSvd — Watermarking Comparison

A test comparison of wtDct vs dwtDctSvd watermarking methods based on the baseline data.

## Embedding Quality

| Method | Avg PSNR (dB) | Avg SSIM | Avg Bit Accuracy |
|--------|--------------|----------|-----------------|
| dwtDct | 37.73 | 0.9766 | 85.9% |
| dwtDctSvd | 37.92 | 0.9848 | 100.0% |

## Interpretation
- PSNR > 35 dB = acceptable visual quality
- SSIM > 0.95 = visually similar to original
- Bit Accuracy > 80% = watermark reliably detectable

## Method Notes
- **dwtDct**: DWT + DCT. Baseline method.
- **dwtDctSvd**: DWT + DCT + SVD. Adds singular value decomposition for stronger robustness, especially against compression attacks.

# dwtDct vs dwtDctSvd — Robustness Comparison

## Watermark Survival Rate by Transformation (%)

| Transform | dwtDct (%) | dwtDctSvd (%) |
|-----------|----------:|----------:|
| jpeg_50      |     56.2 |     91.2 |
| jpeg_70      |     57.8 |     95.0 |
| jpeg_90      |     60.3 |     90.0 |
| crop_25      |     49.7 |     50.9 |
| crop_50      |     50.9 |     60.9 |
| resize_50    |     88.1 |    100.0 |
| resize_75    |     87.5 |    100.0 |
| blur         |     61.3 |    100.0 |
| screenshot   |     85.9 |    100.0 |

## Notes
- ~50% bit accuracy = watermark destroyed (random chance)
- > 80% = watermark survives
- dwtDctSvd expected to outperform dwtDct under JPEG compression

## Summary

dwtDctSvd is the stronger method across nearly every metric. At embedding, it achieves perfect bit accuracy (100% vs 85.9%) with marginally better image quality, suggesting SVD places the watermark more efficiently. The robustness gap is most striking under JPEG compression — the primary transform used by mobile platforms — where dwtDctSvd survives at 90–95% vs dwtDct's 56–60%. dwtDctSvd also achieves 100% survival under blur, resize, and screenshot, compared to 61–88% for dwtDct. The one weakness both methods share is cropping: aggressive cropping (25–50%) destroys both watermarks equally (~50%), which is expected since cropping physically removes the regions where the watermark is embedded. Overall, dwtDctSvd is the clear choice for mobile environments where JPEG compression is unavoidable.