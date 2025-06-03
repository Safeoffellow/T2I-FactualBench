from diffusers import StableDiffusion3Pipeline
import torch

def generate_sd35(prompt: str, output_path: str, device):
    pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3.5-large", torch_dtype=torch.bfloat16).to(device) 
    image = pipe(
        prompt= prompt,
        negative_prompt="",
        num_inference_steps=28,
        height=1024,
        width=1024,
        guidance_scale=3.5,
    ).images[0]
    image.save(output_path)