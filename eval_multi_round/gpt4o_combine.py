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

score_all_type = {"artifact": [], "animal": [], "plant": [], "food": [], "person": [], "event": [], "celestial": [], "location": []}

type_relation = {"animal": "animal", "pet": "animal", "hat": "artifact", "bag": "artifact", "car": "artifact", "clothes": "artifact", "sport_equipment": "artifact", "music_instrument": "artifact", "music": "artifact", "electronic": "artifact", "other": "artifact", "celestial": "celestial", "event": "event", "person": "person", "landmark": "location", "natural_landform": "location", "plant": "plant", "food": "food"}

def infer(result_path, model_name, level, eval_model):
    score_path = os.path.join(result_path, model_name, level+ f"{eval_model}_score")
    json_files = [file for file in os.listdir(score_path) if file.endswith('.jsonl')]
    print(json_files)
    result_txt = os.path.join(result_path, model_name, "result_category.txt")

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

            
        for key, value in score_all_type.items():
            writer.write("\n")
            writer.write(f"{key}_score: {round((sum(value) / (len(value)*4)),3)}")
            writer.write("\n")

        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation Script")
    parser.add_argument("--result_path", type=str, required=True, help="Path to the result directory.")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--level", type=str, required=True, help="Level name for logging purposes.")
    parser.add_argument("--eval_model", type=str, required=True, help="Name of Evaluation Model")
    args = parser.parse_args()

    model_name_list = ["fluxdev", "msdiffusion", "msdiffusion_ref_without", "pixart", "playground", "sd1.5_new", "sd3_real", "sd3.5_new", "sdXL", "ssr_encoder", "ssr_encoder"]

    for model_name in model_name_list:
        if args.level == "level_all":
            infer(args.result_path, model_name, "Knowledge_Momerization", args.eval_model)
            # infer(args.result_path, args.model_name, "Knowledge_Understanding", args.eval_model)
            # infer(args.result_path, args.model_name, "Knowledge_Applying", args.eval_model)
        else:
            infer(args.result_path, args.model_name, args.level, args.eval_model)