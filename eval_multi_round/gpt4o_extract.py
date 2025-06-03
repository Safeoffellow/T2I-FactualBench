import requests
import json
import jsonlines
import os
import re
import time
import logging
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import glob

def extract_concept_rating(text):
    # 使用正则表达式查找分数
    match = re.search(r'Total Rating:\s*(\d+)', text)
    if match:
        return int(match.group(1))
    else:
        return None  # 如果没有找到分数，返回None

def extract_integration_rating(text):
    # 注意正则表达式的模式
    match = re.search(r'Total Rating:\s*(\d+)', text)
    if match:
        return int(match.group(1))
    else:
        return None  # 如果没有找到分数，返回None

def extract_task_score(text):
    match = re.search(r'Task Score:\s*(\d+)', text)
    if match:
        return int(match.group(1))
    else:
        return None


def infer(result_path, model_name, level, eval_model):
    response_path = os.path.join(result_path, model_name, level+ f"_{eval_model}")
    print(response_path)
    score_path = os.path.join(result_path, model_name, level+ f"{eval_model}_score")
    os.makedirs(score_path, exist_ok=True)
    json_files = [file for file in os.listdir(response_path) if file.endswith('.jsonl')]
    print(json_files)
    for json_name in json_files:
        response_file = os.path.join(response_path, json_name)
        with jsonlines.open(response_file, 'r') as reader:
            score_file = os.path.join(score_path, json_name)
            with jsonlines.open(score_file, 'w') as writer:
                for line in reader:
                    concept_score_list = []
                    for concept_score in line["concept_score"]:
                        #print(part["gpt4o_reasponse"])
                        concept_score_list.append(extract_concept_rating(concept_score["gpt4o_reasponse"]))
                    if line["task_score"] != "":
                        task_score = extract_task_score(line["task_score"])
                    else:
                        task_score = -1
                    
                    if line["integration_score"] != "":
                        integration_score = extract_integration_rating(line["integration_score"])
                    else:
                        integration_score = -1
                    # print("Line:", line["whole"])
                    print("concept_score:", concept_score_list)
                    print("task_score:", task_score)
                    print("integration_score", integration_score)
                    line["score"] = {"concept_score": concept_score_list, "task_score": task_score, "integration_score": integration_score}
                    writer.write(line)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation Script")
    parser.add_argument("--result_path", type=str, required=True, help="Path to the result directory.")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--level", type=str, required=True, help="Level name for logging purposes.")
    parser.add_argument("--eval_model", type=str, required=True, help="Name of Evaluation Model")
    args = parser.parse_args()
    
    if args.level == "level_all":
        infer(args.result_path, args.model_name, "Knowledge_Momerization", args.eval_model)
        infer(args.result_path, args.model_name, "Knowledge_Understanding", args.eval_model)
        infer(args.result_path, args.model_name, "Knowledge_Applying", args.eval_model)
        infer(args.result_path, args.model_name, "Knowledge_Memorization_add", args.eval_model)
    else:
        # models = ["fluxdev", "msdiffusion", "pixart", "playground", "sd1.5_new", "sd3_real", "sd3.5_new", "sdXL", "ssr_encoder"]
        for model in models:
            infer(args.result_path, model, args.level, args.eval_model)