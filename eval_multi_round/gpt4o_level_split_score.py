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

score_list = {"fluxdev": [], "msdiffusion":[], "pixart": [], "playground":[], "sd1.5_new":[], "sd3.5_new":[], "sdXL":[], "ssr_encoder":[], "dalle3":[], "sd3.5_new_text_knowledge":[], "fluxdev_text_knowledge":[]}

def infer(result_path, model_name, level, eval_model):
    if level == "Knowledge_Understanding" or level == "Knowledge_Applying":
        score_path = os.path.join(result_path, model_name, level+ f"{eval_model}_score")
        json_files = [file for file in os.listdir(score_path) if file.endswith('.jsonl')]
        # knowledge_memorization_add
        print(json_files)
        for json_name in json_files:
            score_file = os.path.join(score_path, json_name)
            data_all = []
            with jsonlines.open(score_file, 'r') as reader:
                for line in reader:
                    for concept in line["reference_image"]: 
                        if concept not in score_list[model_name]:
                            score_list[model_name].append(concept)

    if level == "Knowledge_Momerization":
        print(model_name)
        result_txt = os.path.join(result_path, model_name, "result_score_split_2.txt")
        dict_score = {}
        score_all = []
        json_files = [os.path.join(result_path, model_name, "Knowledge_Momerization"+ f"{eval_model}_score", "knowledge_momerization.jsonl")]
        # print(score_list[model_name])
        with open(result_txt, 'w') as writer:
            # metric_all = []
            # concept_all = []
            # task_all = []
            # integration_all = []
            for json_name in json_files:
                with jsonlines.open(json_name, 'r') as reader:
                    for line in reader:
                        # print(line["concept"])
                        concept_under = line["reference_image"][0]
                        score_temp = line["score"]["concept_score"][0]
                        dict_score[concept_under] = score_temp
                
                # print(score_list[model_name])
                for concept in score_list[model_name]:
                    if concept not in dict_score.keys():
                        score_all.append(0)
                    elif dict_score[concept] == None:
                        score_all.append(0)
                    else:
                        # print(concept)
                        score_all.append(dict_score[concept])

                print(sum(score_all))
                writer.write("\n")
                writer.write(f"All_score: {round((sum(score_all) / (len(score_all)*4)),3)}")
                writer.write("\n")

        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation Script")
    parser.add_argument("--result_path", type=str, required=True, help="Path to the result directory.")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--level", type=str, required=True, help="Level name for logging purposes.")
    parser.add_argument("--eval_model", type=str, required=True, help="Name of Evaluation Model")
    args = parser.parse_args()

    model_name_list = ["fluxdev", "msdiffusion", "pixart", "playground", "sd1.5_new", "sd3.5_new", "sdXL", "ssr_encoder", "sd3.5_new_text_knowledge", "dalle3", "fluxdev_text_knowledge"]

    for model_name in model_name_list:
        if args.level == "level_all":
            infer(args.result_path, model_name, "Knowledge_Understanding", args.eval_model)
            # infer(args.result_path, model_name, "Knowledge_Applying", args.eval_model)
            infer(args.result_path, model_name, "Knowledge_Momerization", args.eval_model)
        else:
            infer(args.result_path, args.model_name, args.level, args.eval_model)