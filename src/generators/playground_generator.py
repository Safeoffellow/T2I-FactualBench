from diffusers import DiffusionPipeline
import torch

def generate_playground(prompt: str, output_path: str, device):
    pipe = DiffusionPipeline.from_pretrained(
        "playgroundai/playground-v2.5-1024px-aesthetic",
        torch_dtype=torch.float16,
        variant="fp16").to(device) 
    image = pipe(
        prompt= prompt,
    ).images[0]
    image.save(output_path)