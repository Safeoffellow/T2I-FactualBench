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

def extract_concept_rating_dimension(text):
    try:
        # 定义正则表达式模式
        pattern = r"(Shape Accuracy|Color Accuracy|Texture Representation|Feature Details): (\d+)(/\d+)?"
        # 查找所有匹配的部分
        matches = re.findall(pattern, text)
        if not matches:
            raise ValueError("No matches found in the provided text.")
        # 提取结果到字典，并考虑可能的异常情况
        results = []
        for match in matches:
            field_name, score, _ = match
            if field_name in results:
                raise ValueError(f"Duplicate entry found for {field_name}.")
            results.append(int(score))

        return results

    except re.error as e:
        print(f"Regex error: {e}")
    except ValueError as ve:
        print(f"Value error: {ve}")
    except Exception as ex:
        print(f"An unexpected error occurred: {ex}")

def extract_integration_rating(text):
    # 注意正则表达式的模式
    match = re.search(r'Total Rating:\s*(\d+)', text)
    if match:
        return int(match.group(1))
    else:
        return None  # 如果没有找到分数，返回None

def extract_integration_rating_dimension(text):
    try:
        # 定义正则表达式模式
        pattern = r"(Overall Harmony|Visual Completeness|Authenticity|Prompt Following): (\d+)(/\d+)?"
        # 查找所有匹配的部分
        matches = re.findall(pattern, text)
        if not matches:
            raise ValueError("No matches found in the provided text.")
        # 提取结果到字典，并考虑可能的异常情况
        results = []
        for match in matches:
            field_name, score, _ = match
            if field_name in results:
                raise ValueError(f"Duplicate entry found for {field_name}.")
            results.append(int(score))

        return results

    except re.error as e:
        print(f"Regex error: {e}")
    except ValueError as ve:
        print(f"Value error: {ve}")
    except Exception as ex:
        print(f"An unexpected error occurred: {ex}")

def extract_task_score(text):
    match = re.search(r'Task Score:\s*(\d+)', text)
    if match:
        return int(match.group(1))
    else:
        return None


def infer(result_path, model_name, level, eval_model):
    response_path = os.path.join(result_path, model_name, level+ f"_{eval_model}")
    print(response_path)
    score_path = os.path.join(result_path, model_name, level+ f"{eval_model}_score_dimension")
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
                    concept_score_dimenstion = []
                    for concept_score in line["concept_score"]:
                        #print(part["gpt4o_reasponse"])
                        concept_score_list.append(extract_concept_rating(concept_score["gpt4o_reasponse"]))
                        concept_score_dimenstion.append(extract_concept_rating_dimension(concept_score["gpt4o_reasponse"]))
                    if line["task_score"] != "":
                        task_score = extract_task_score(line["task_score"])
                    else:
                        task_score = -1
                    if line["integration_score"] != "":
                        integration_score = extract_integration_rating(line["integration_score"])
                        integration_score_dimension = extract_integration_rating_dimension(line["integration_score"])
                    else:
                        integration_score = -1
                        integration_score_dimension = []
                    # print("Line:", line["whole"])
                    print("concept_score:", concept_score_list)
                    print("task_score:", task_score)
                    print("integration_score", integration_score)
                    line["score"] = {"concept_score": concept_score_list, "concept_score_dimenstion": concept_score_dimenstion, "task_score": task_score, "integration_score": integration_score, "integration_dimension": integration_score_dimension}
                    writer.write(line)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation Script")
    parser.add_argument("--result_path", type=str, required=True, help="Path to the result directory.")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--level", type=str, required=True, help="Level name for logging purposes.")
    parser.add_argument("--eval_model", type=str, required=True, help="Name of Evaluation Model")
    args = parser.parse_args()
    
    models = ["fluxdev", "msdiffusion", "pixart", "playground", "sd1.5_new", "dalle3", "sd3.5_new", "sdXL", "ssr_encoder", "sd3.5_new_text_knowledge", "fluxdev_text_knowledge"]
    for model in models:
        if args.level == "level_all":
            infer(args.result_path, model, "Knowledge_Momerization", args.eval_model)
            infer(args.result_path, model, "Knowledge_Understanding", args.eval_model)
            infer(args.result_path, model, "Knowledge_Applying", args.eval_model)
            infer(args.result_path, model, "Knowledge_Memorization_add", args.eval_model)