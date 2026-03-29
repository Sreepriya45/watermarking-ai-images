"""
Baseline: Generate 10 images only for testing pipeline
"""
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import os
from tqdm import tqdm

# ── Detect best device ─────────────────────────────
if torch.cuda.is_available():
    device = "cuda"
    dtype = torch.float16
elif torch.backends.mps.is_available():
    device = "mps"       # Apple Silicon — much faster!
    dtype = torch.float32
else:
    device = "cpu"
    dtype = torch.float32

print(f"Using device: {device}")

# ── Only 2 prompts x 5 images = 10 total ──────────
PROMPTS = [
    "a realistic portrait of a person, "
    "natural lighting, high quality photo",
    "a photo of a university campus, "
    "sunny day, realistic",
]
NUM_PER_PROMPT = 5   # 10 images total
STEPS = 15           # was 30 — 2x faster

# ── Load model ─────────────────────────────────────
print("Loading model...")
pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=dtype,
    safety_checker=None
)
pipe = pipe.to(device)
pipe.enable_attention_slicing()  # saves memory

os.makedirs("data/original", exist_ok=True)

# ── Generate ───────────────────────────────────────
count = 0
for i, prompt in enumerate(PROMPTS):
    for j in tqdm(range(NUM_PER_PROMPT),
                  desc=f"Prompt {i+1}/{len(PROMPTS)}"):
        image = pipe(
            prompt,
            num_inference_steps=STEPS,
            guidance_scale=7.5,
            height=512,
            width=512
        ).images[0]

        path = f"data/original/img_{i:02d}_{j:02d}.png"
        image.save(path)
        count += 1
        print(f"  Saved {path}")

print(f"\nDone! {count} images saved to data/original/")