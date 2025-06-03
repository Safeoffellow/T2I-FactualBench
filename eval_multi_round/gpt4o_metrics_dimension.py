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
    score_path = os.path.join(result_path, model_name, level+ f"{eval_model}_score_dimension")
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
        concept_dimension_all_1 = []
        concept_dimension_all_2 = []
        concept_dimension_all_3 = []
        concept_dimension_all_4 = []
        task_all = []
        integration_all = []
        integration_dimension_all_1 = []
        integration_dimension_all_2 = []
        integration_dimension_all_3 = []
        integration_dimension_all_4 = []

        for json_name in json_files:
            score_file = os.path.join(score_path, json_name)
            data_all = []

            with open(score_file, 'r', encoding='utf-8') as reader:
                for line in reader:
                    data_all.append(json.loads(line))
            

            concept_score = 0
            concept_score_1 = 0
            concept_score_2 = 0
            concept_score_3 = 0
            concept_score_4 = 0
            task_score = 0
            integration_score = 0
            integration_score_1 = 0
            integration_score_2 = 0
            integration_score_3 = 0
            integration_score_4 = 0
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
                for score in line["score"]["concept_score_dimenstion"]:
                    if score is None:
                        score = [0,0,0,0]
                    if len(score) == 4:
                        print(score, index)
                        concept_score_1 += score[0]
                        concept_score_2 += score[1]
                        concept_score_3 += score[2]
                        concept_score_4 += score[3]


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
                if line["score"]["integration_dimension"] is None:
                    line["score"]["integration_dimension"] = [0,0,0,0]
                if len(line["score"]["integration_dimension"]) != 4:
                    # 对 None 值的处理，例如使用默认值，跳过，或抛出异常
                    line["score"]["integration_dimension"] = [0,0,0,0]
                integration_score_1 += line["score"]["integration_dimension"][0]
                integration_score_2 += line["score"]["integration_dimension"][1]
                integration_score_3 += line["score"]["integration_dimension"][2]
                integration_score_4 += line["score"]["integration_dimension"][3]
            
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
            Metric_concept_shape = round(concept_score_1/ (len(data_all) * concept_num),3)
            Metric_concept_color = round(concept_score_2/ (len(data_all) * concept_num),3)
            Metric_concept_texture = round(concept_score_3/ (len(data_all) * concept_num),3)
            Metric_concept_detail = round(concept_score_4/ (len(data_all) * concept_num),3)
            writer.write(f"Metric concept shape: {Metric_concept_shape}")
            writer.write("\n")
            writer.write(f"Metric concept color: {Metric_concept_color}")
            writer.write("\n")
            writer.write(f"Metric concept texture: {Metric_concept_texture}")
            writer.write("\n")
            writer.write(f"Metric concept detail: {Metric_concept_detail}")
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
            
            if integration_score_1 != 0:
                Metric_integration_seamless = round(integration_score_1 / (len(data_all)), 3)
                Metric_integration_completeness = round(integration_score_2 / (len(data_all)), 3)
                Metric_integration_authenticity = round(integration_score_3 / (len(data_all)), 3)
                Metric_integration_prompt = round(integration_score_4 / (len(data_all)), 3)
                writer.write(f"Metric Integration seamless: {Metric_integration_seamless}")
                writer.write("\n")
                writer.write(f"Metric Integration completeness: {Metric_integration_completeness}")
                writer.write("\n")
                writer.write(f"Metric Integration authenticity: {Metric_integration_authenticity}")
                writer.write("\n")
                writer.write(f"Metric Integration prompt: {Metric_integration_prompt}")
                writer.write("\n")
                integration_dimension_all_1.append(Metric_integration_seamless)
                integration_dimension_all_2.append(Metric_integration_completeness)
                integration_dimension_all_3.append(Metric_integration_authenticity)
                integration_dimension_all_4.append(Metric_integration_prompt)

            if level == "Knowledge_Momerization":
                Metric_all = Metric_concept
            elif level == "Knowledge_Understanding":
                Metric_all = round((Metric_concept + Metric_task) / 2, 3)
            elif level == "Knowledge_Applying":
                Metric_all = round((Metric_concept + Metric_task + Metric_integration) / 3, 3)
            elif level == "Knowledge_Memorization_add":
                Metric_all = Metric_concept
            else:
                Metric_all = "Wrong Level!"

            writer.write(f"Metric All: {Metric_all}")
            writer.write("\n")
            writer.write("*" * 40)
            writer.write("\n")
            concept_all.append(Metric_concept)
            concept_dimension_all_1.append(Metric_concept_shape)
            concept_dimension_all_2.append(Metric_concept_color)
            concept_dimension_all_3.append(Metric_concept_texture)
            concept_dimension_all_4.append(Metric_concept_detail)
            # task_all.append(Metric_task)
            # integration_all.append(Metric_integration)
            metric_all.append(Metric_all)

        writer.write(f"Level Metric Concept: {round((sum(concept_all) / len(json_files)),3)*100}")
        writer.write("\n")
        writer.write(f"Level Metric Concept shape: {round((sum(concept_dimension_all_1) / len(json_files)),3)*100}")
        writer.write("\n")
        writer.write(f"Level Metric Concept color: {round((sum(concept_dimension_all_2) / len(json_files)),3)*100}")
        writer.write("\n")
        writer.write(f"Level Metric Concept texture: {round((sum(concept_dimension_all_3) / len(json_files)),3)*100}")
        writer.write("\n")
        writer.write(f"Level Metric Concept detail: {round((sum(concept_dimension_all_4) / len(json_files)),3)*100}")
        writer.write("\n")
        if level == "Knowledge_Understanding" or level == "Knowledge_Applying":
            writer.write(f"Level Metric Task: {round((sum(task_all) / len(json_files)),3)}")
            writer.write("\n")
        if level == "Knowledge_Applying":
            writer.write(f"Level Metric Integration: {round((sum(integration_all) / len(json_files)),3)}")
            writer.write("\n")
            writer.write("\n")
            writer.write(f"Level Metric Integration seamless: {round((sum(integration_dimension_all_1) / len(json_files)),3)*100}")
            writer.write("\n")
            writer.write(f"Level Metric Integration completeness: {round((sum(integration_dimension_all_2) / len(json_files)),3)*100}")
            writer.write("\n")
            writer.write(f"Level Metric Integration authenticity: {round((sum(integration_dimension_all_3) / len(json_files)),3)*100}")
            writer.write("\n")
            writer.write(f"Level Metric Integration Prompt: {round((sum(integration_dimension_all_4) / len(json_files)),3) *100}")
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
    models = ["fluxdev", "msdiffusion", "pixart", "playground", "sd1.5_new", "dalle3", "sd3.5_new", "sdXL", "ssr_encoder", "sd3.5_new_text_knowledge", "fluxdev_text_knowledge"]
    for model in models:
        if args.level == "level_all":
            infer(args.result_path, model, "Knowledge_Momerization", args.eval_model)
            infer(args.result_path, model, "Knowledge_Understanding", args.eval_model)
            infer(args.result_path, model, "Knowledge_Applying", args.eval_model)
            infer(args.result_path, model, "Knowledge_Memorization_add", args.eval_model)
        else:
            infer(args.result_path, args.model_name, args.level, args.eval_model)