from diffusers import StableDiffusionPipeline
import torch

def generate_sd15(prompt: str, output_path: str, device):
    pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16).to(device)
    with autocast():
        image = pipe(
            prompt= prompt,
            negative_prompt="",
        ).images[0]
        image.save(output_path)
