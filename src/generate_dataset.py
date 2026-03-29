"""
Step 1: Generate AI image dataset using Stable Diffusion
"""
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import os
from tqdm import tqdm

# ── Config ─────────────────────────────────────────
OUTPUT_DIR   = "dataset/original"
NUM_PER_PROMPT = 10
IMAGE_SIZE   = 512

PROMPTS = [
    "a realistic portrait of a young woman, "
    "natural lighting, high quality photo",
    "a realistic portrait of an elderly man, "
    "natural lighting, high quality photo",
    "a professional headshot of a person, "
    "studio lighting, sharp focus",
    "a photo of a coffee cup on a wooden table, "
    "natural lighting, realistic",
    "a photo of a smartphone on a clean desk, "
    "product photography",
    "a photo of a university campus building, "
    "sunny day, realistic photography",
    "a photo of a busy city street, "
    "people walking, realistic",
]

def generate_dataset():
    device = "cuda" if torch.cuda.is_available() \
             else "cpu"
    print(f"Using device: {device}")

    # Load model
    print("Loading Stable Diffusion...")
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16 if device == "cuda"
                    else torch.float32,
        safety_checker=None
    )
    pipe = pipe.to(device)
    pipe.enable_attention_slicing()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    total = len(PROMPTS) * NUM_PER_PROMPT
    print(f"Generating {total} images...")

    img_count = 0
    for i, prompt in enumerate(PROMPTS):
        for j in tqdm(range(NUM_PER_PROMPT),
                      desc=f"Prompt {i+1}/{len(PROMPTS)}"):
            image = pipe(
                prompt,
                num_inference_steps=30,
                guidance_scale=7.5,
                height=IMAGE_SIZE,
                width=IMAGE_SIZE
            ).images[0]

            filename = f"img_{i:02d}_{j:02d}.png"
            save_path = os.path.join(OUTPUT_DIR, filename)
            image.save(save_path)
            img_count += 1

    print(f"\nDone! Generated {img_count} images")
    print(f"Saved to: {OUTPUT_DIR}")

if __name__ == "__main__":
    generate_dataset()