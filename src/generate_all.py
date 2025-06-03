import os
import torch
import argparse
from tqdm import tqdm
from generators import (
    generate_sdxl,
    generate_sd15,
    generate_sd35,
    generate_fluxdev,
    generate_pixart,
    generate_playground,
    generate_dalle3,
)
import torch.distributed as dist
import deepspeed
import datetime
import random
import glob
import jsonlines

MODEL_GENERATORS = {
    "sd15": generate_sd15,
    "sdxl": generate_sdxl,
    "sd35": generate_sd35,
    "fluxdev": generate_fluxdev,
    "pixart": generate_pixart,
    "playground": generate_playground,
    "dalle3": generate_dalle3,
}

def same_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_directories(dirs):
    if dist.get_rank() == 0:
        for dir in dirs:
            os.makedirs(dir, exist_ok=True)
    dist.barrier()

def generate_all(model_name, prompt_base, output_root, level, knowledge_injection):
    same_seed(42)
    if knowledge_injection not in ["None", "text"]:
        raise ValueError(f"Invalid knowledge_injection: {knowledge_injection}. Choose from ['None', 'text']")
    if model_name not in MODEL_GENERATORS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODEL_GENERATORS.keys())}")

    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        rank = 0
        world_size = 1
        local_rank = 0
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(local_rank)

    if knowledge_injection == "text":
        output_dir = os.path.join(output_root, model_name+"_text", level)
    else:
        output_dir = os.path.join(output_root, model_name, level)
    prompt_base = os.path.join(prompt_base, level)

    json_files = glob.glob(os.path.join(prompt_base, '*.jsonl'))

    generator_fn = MODEL_GENERATORS[model_name]
    
    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)
    dist.barrier()

    for json_file in json_files:
        prompt_name = os.path.basename(json_file).replace(".jsonl", "")
        out_image_dir = os.path.join(output_dir, prompt_name + "_image")
        create_directories([out_image_dir])
        
        local_reference_list = []
        json_output_path = os.path.join(output_dir, prompt_name + ".jsonl")
        with jsonlines.open(json_file, "r") as reader:
            lines = list(reader)
        total_lines = len(lines)
        

        for index, line in tqdm(enumerate(lines), total=total_lines, desc=f"{prompt_name} Processing lines", leave=False):
            if index % world_size != rank:
                continue
            if knowledge_injection == "None":
                prompt = line["sentence"]
            elif knowledge_injection == "text":
                prompt = line["sentence_text_knowledge"]
            out_image_path = os.path.join(out_image_dir, f'{index}.png')

            kwargs = {"device": device} if model_name != "dalle3" else {}
            generator_fn(prompt, out_image_path, **kwargs)

            line["result_image"] = out_image_path
            local_reference_list.append(line)
                

        all_reference_lists = [None] * world_size
        dist.all_gather_object(all_reference_lists, local_reference_list)

        # Only rank 0 writes to the JSON file
        if rank == 0:
            # Combine all lists
            combined_reference_list = []
            for ref_list in all_reference_lists:
                combined_reference_list.extend(ref_list)

            with jsonlines.open(json_output_path, "w") as writer:
                for line in combined_reference_list:
                    writer.write(line)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model name (e.g., sd35)")
    parser.add_argument("--prompt_dir", type=str, required=True, help="Directory containing prompts by level")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save generated images")
    parser.add_argument("--level", type=str, required=True, choices=["SKCM", "SKCI", "MKCC", "level_all"], help="Task level name: SKCM, SKCI, MKCC, level_all")
    parser.add_argument("--knowledge_injection", type=str, required=True, choices=["None", "text"], help="One of: None, text (for knowledge-enhanced prompts)")
    args = parser.parse_args()
    print("Initializing process group...")
    dist.init_process_group(backend="nccl", init_method="env://", timeout=datetime.timedelta(minutes=5))
    deepspeed.init_distributed()
    if args.level == "level_all":
        generate_all(args.model, args.prompt_dir, args.output_dir, "SKCM", args.knowledge_injection)
        generate_all(args.model, args.prompt_dir, args.output_dir, "SKCI", args.knowledge_injection)
        generate_all(args.model, args.prompt_dir, args.output_dir, "MKCC", args.knowledge_injection)
    else:
        generate_all(args.model, args.prompt_dir, args.output_dir, args.level, args.knowledge_injection)
