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

def infer(result_path, model_name, level, eval_model):
    score_path = os.path.join(result_path, model_name, level+ f"{eval_model}_score")
    json_files = [file for file in os.listdir(score_path) if file.endswith('.jsonl')]
    print(json_files)
    result_txt = os.path.join(score_path, "result.txt")

    with open(result_txt, 'w') as writer:
        # common_metric = []
        # common_concept_metric = []
        # common_elements = []
        # common_prompt_following = []
        # uncommon_metric = []
        # uncommon_concept_metric = []
        # uncommon_elements = []
        # uncommon_prompt_following =[]

        metric_all = []
        concept_all = []
        task_all = []
        integration_all = []

        for json_name in json_files:
            score_file = os.path.join(score_path, json_name)
            data_all = []

            with open(score_file, 'r', encoding='utf-8') as reader:
                for line in reader:
                    data_all.append(json.loads(line))
            

            concept_score = 0
            task_score = 0
            integration_score = 0
            concept_num = 0

            for index, line in enumerate(data_all):
                concept_num = len(line["score"]["concept_score"])
                for score in line["score"]["concept_score"]:
                    if score is not None:
                        concept_score += score
                    else:
                        # 对 None 值的处理，例如使用默认值，跳过，或抛出异常
                        print(index+1)
                        print("Warning: `score` is None, skipping this addition.")
                        score = 0
                        concept_score += score

                if line["score"]["task_score"] != -1:
                    if line["score"]["task_score"] is not None:
                        task_score += line["score"]["task_score"] * 4
                    else:
                        # 对 None 值的处理，例如使用默认值，跳过，或抛出异常
                        print(index+1)
                        print("Warning: `score` is None, skipping this addition.")
                        score = 0
                        task_score += score
                else:
                    task_score = -1
    
                if line["score"]["integration_score"] != -1:
                    if line["score"]["integration_score"] is not None:
                        integration_score += line["score"]["integration_score"]
                    else:
                        # 对 None 值的处理，例如使用默认值，跳过，或抛出异常
                        print(index+1)
                        print("Warning: `score` is None, skipping this addition.")
                        score = 0
                        integration_score += score
                else:
                    integration_score = -1
            
            print("concept_num:", concept_num)
            writer.write("*" * 40)
            writer.write("\n")
            writer.write(json_name)
            writer.write("\n")
            writer.write(f"Total Concept: {concept_score}")
            writer.write("\n")

            if task_score != -1:
                writer.write(f"Total Task: {task_score}")
                writer.write("\n")
            if integration_score != -1:
                writer.write(f"Total Integration: {integration_score}")
                writer.write("\n")

            Metric_concept = round(concept_score/ (len(data_all) * concept_num * 4),3)
            writer.write(f"Metric concept: {Metric_concept}")
            writer.write("\n")

            if task_score != -1:
                Metric_task = round(task_score / (len(data_all) * 4), 3)
                writer.write(f"Metric Task: {Metric_task}")
                task_all.append(Metric_task)
                writer.write("\n")

            if integration_score != -1:
                Metric_integration = round(integration_score / (len(data_all) * 4), 3)
                writer.write(f"Metric Integration: {Metric_integration}")
                integration_all.append(Metric_integration)
                writer.write("\n")

            if level == "Knowledge_Momerization":
                Metric_all = Metric_concept
            elif level == "Knowledge_Understanding":
                Metric_all = round((Metric_concept + Metric_task) / 2, 3)
            elif level == "Knowledge_Applying":
                Metric_all = round((Metric_concept + Metric_task + Metric_integration) / 3, 3)
            else:
                Metric_all = "Wrong Level!"

            writer.write(f"Metric All: {Metric_all}")
            writer.write("\n")
            writer.write("*" * 40)
            writer.write("\n")
            concept_all.append(Metric_concept)
            # task_all.append(Metric_task)
            # integration_all.append(Metric_integration)
            metric_all.append(Metric_all)

        writer.write(f"Level Metric Concept: {round((sum(concept_all) / len(json_files)),3)}")
        writer.write("\n")
        if level == "Knowledge_Understanding" or level == "Knowledge_Applying":
            writer.write(f"Level Metric Task: {round((sum(task_all) / len(json_files)),3)}")
            writer.write("\n")
        if level == "Knowledge_Applying":
            writer.write(f"Level Metric Integration: {round((sum(integration_all) / len(json_files)),3)}")
            writer.write("\n")
        
        print(metric_all)
        writer.write(f"Level Metric All: {round((sum(metric_all) / len(json_files)),3)}")
        writer.write("\n")
            

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
    else:
        infer(args.result_path, args.model_name, args.level, args.eval_model)