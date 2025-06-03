from diffusers import PixArtAlphaPipeline
import torch

def generate_pixart(prompt: str, output_path: str, device):
    pipe = PixArtAlphaPipeline.from_pretrained("PixArt-alpha/PixArt-XL-2-1024-MS", torch_dtype=torch.float16).to(device)
    image = pipe(
        prompt= prompt,
    ).images[0]
    image.save(output_path)