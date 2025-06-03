from diffusers import StableDiffusionXLPipeline
import torch

def generate_sdxl(prompt: str, output_path: str, device):
    pipe = StableDiffusionXLPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16).to(device)
    image = pipe(
        prompt= prompt
    ).images[0]
    image.save(output_path)