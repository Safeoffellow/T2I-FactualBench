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



type_relation = {"animal": "animal", "pet": "animal", "hat": "artifact", "bag": "artifact", "car": "artifact", "clothes": "artifact", "sport_equipment": "artifact", "music_instrument": "artifact", "music": "artifact", "electronic": "artifact", "other": "artifact", "celestial": "celestial", "event": "event", "person": "person", "landmark": "location", "natural_landform": "location", "plant": "plant", "food": "food", "artifact": "artifact", "location": "location", "aircraft": "artifact"}

def infer(result_path, model_name, level, eval_model):
    score_all_type = {"artifact": [], "animal": [], "plant": [], "food": [], "person": [], "event": [], "celestial": [], "location": []}
    score_path = os.path.join(result_path, model_name, level+ f"{eval_model}_score")
    json_files = []
    json_file = os.path.join(result_path, model_name, "Knowledge_Momerization"+ f"{eval_model}_score", "knowledge_momerization.jsonl")
    json_files.append(json_file)
    # knowledge_memorization_add
    json_file_add = os.path.join(result_path, model_name, "Knowledge_Memorization_add"+ f"{eval_model}_score", "knowledge_momerization_add.jsonl")
    json_files.append(json_file_add)
    print(json_files)
    result_txt = os.path.join(result_path, model_name, "result_category_new.txt")

    with open(result_txt, 'w') as writer:
        # metric_all = []
        # concept_all = []
        # task_all = []
        # integration_all = []
        for json_name in json_files:
            score_file = os.path.join(score_path, json_name)
            data_all = []

            with open(score_file, 'r', encoding='utf-8') as reader:
                for line in reader:
                    data_all.append(json.loads(line))
            

            concept_score = 0
            # task_score = 0
            # integration_score = 0
            # concept_num = 0

            for index, line in enumerate(data_all):
                concept_num = len(line["score"]["concept_score"])
                for score in line["score"]["concept_score"]:
                    if score is not None:
                        score_all_type[type_relation[line["type"][0]]].append(score)
                    else:
                        # 对 None 值的处理，例如使用默认值，跳过，或抛出异常
                        print(index+1)
                        print("Warning: `score` is None, skipping this addition.")
                        score = 0
                        score_all_type[type_relation[line["type"][0]]].append(score)

        all_num = []
        score_all = []
        if model_name == "dalle3":
            for key, value in score_all_type.items():
                if key == "person":
                    continue
                writer.write("\n")
                writer.write(f"{key}_num: {len(value)}")
                all_num.append(len(value))
                score_all.append(sum(value))
                writer.write("\n")
                writer.write(f"{key}_score: {round((sum(value) / (len(value)*4)),3)}")
                writer.write("\n")
            writer.write("\n")
            writer.write(f"Total Score: {round((sum(score_all) / (sum(all_num)*4)),3)}")
        else:
            for key, value in score_all_type.items():
                writer.write("\n")
                writer.write(f"{key}_num: {len(value)}")
                all_num.append(len(value))
                score_all.append(sum(value))
                writer.write("\n")
                writer.write(f"{key}_score: {round((sum(value) / (len(value)*4)),3)}")
                writer.write("\n")
            writer.write("\n")
            writer.write(f"Total Score: {round((sum(score_all) / (sum(all_num)*4)),3)}")

        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation Script")
    parser.add_argument("--result_path", type=str, required=True, help="Path to the result directory.")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--level", type=str, required=True, help="Level name for logging purposes.")
    parser.add_argument("--eval_model", type=str, required=True, help="Name of Evaluation Model")
    args = parser.parse_args()

    # model_name_list = ["fluxdev", "msdiffusion", "pixart", "playground", "sd1.5_new", "sd3_real", "sd3.5_new", "sdXL", "ssr_encoder", "dalle3"]
    model_name_list = ["sd3.5_new_text_knowledge", "fluxdev_text_knowledge"]

    for model_name in model_name_list:
        if args.level == "level_all":
            infer(args.result_path, model_name, "Knowledge_Momerization", args.eval_model)
            # infer(args.result_path, args.model_name, "Knowledge_Understanding", args.eval_model)
            # infer(args.result_path, args.model_name, "Knowledge_Applying", args.eval_model)
        else:
            infer(args.result_path, args.model_name, args.level, args.eval_model)