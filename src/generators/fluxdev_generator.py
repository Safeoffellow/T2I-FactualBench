from diffusers import FluxPipeline
import torch

def generate_fluxdev(prompt: str, output_path: str, device):
    pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16).to(device)
    pipe.enable_model_cpu_offload()
    image = pipe(
        prompt,
        height=1024,
        width=1024,
        guidance_scale=3.5,
        output_type="pil",
        num_inference_steps=50,
        max_sequence_length=512,
    ).images[0]
    image.save(output_path)