import os
import json
import glob
import argparse
import logging
from tool.dino import multi_gpu_eval_dino_score, Dinov2Score
from tool.clip import multi_gpu_eval_clipt_score, multi_gpu_eval_clipi_score, CLIPScore
from datetime import datetime
from accelerate import PartialState
import torch


# 设置日志记录
def setup_logging(level: str, model: str):
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"../log_xingyun/{model}"

    # 如果日志目录不存在，则创建
    try:
        os.makedirs(log_dir)
    except FileExistsError:
        pass

    log_filename = f"{log_dir}/{level}_{current_time}_evaluation.log"
    logging.basicConfig(filename=log_filename, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def evaluate_scores(result_path, level, model):
    setup_logging(level, model)
    logging.info(f"Starting evaluation for level: {level}")

    result_path_json = os.path.join(result_path, level)
    json_files = glob.glob(os.path.join(result_path_json, "*.jsonl"))
    print(result_path_json)
    for json_file in json_files:
        # if "object" not in json_file:
        #     continue
        data_all = []
        with jsonlines.open(json_file, "r") as reader:
            for line in reader:
                data_all.append(line)

        result_images = []
        reference_images = []
        texts = []

        for index, data in enumerate(data_all):
            # if index > 10:
            #     break
            result_images.append(data["result_image"])
            reference_images.append(data["reference_image"])
            texts.append(data["sentence"])

        json_name = json_file.split("/")[-1].replace(".jsonl", "")
        # Compute scores
        # print(json_name)
        # print("参考图个数:", len(reference_images))
        # print(reference_images)
        dino_score = multi_gpu_eval_dino_score(reference_images, result_images, distributed_state=distributed_state)
        clipt_score_value = multi_gpu_eval_clipt_score(texts, result_images, distributed_state=distributed_state)
        clipi_score_value = multi_gpu_eval_clipi_score(reference_images, result_images, distributed_state=distributed_state)
        # Log the scores
        logging.info(f"Level: {level}")
        logging.info(f"Processed file: {json_name}")
        logging.info(f"dino_score: {dino_score}")
        logging.info(f"clipt_score: {clipt_score_value}")
        logging.info(f"clipi_score: {clipi_score_value}")
    logging.info(f"Processed common_score_total: {sum(score_all_common)/len(score_all_common)}")
    logging.info(f"Processed uncommon_score_total: {sum(score_all_uncommon)/len(score_all_uncommon)}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation Script")
    parser.add_argument("--result_path", type=str, required=True, help="Path to the result directory.")
    parser.add_argument("--level", type=str, required=True, help="Level name for logging purposes.")
    parser.add_argument("--model", type=str, required=True, help="Model to generate the image.")
    args = parser.parse_args()

    # Initialize distributed state
    distributed_state = PartialState()
    print("Current working directory:", os.getcwd())
    if args.level == "level_all":
        evaluate_scores(args.result_path, "Knowledge_Momerization", args.model)
        evaluate_scores(args.result_path, "Knowledge_Understanding", args.model)
        evaluate_scores(args.result_path, "Knowledge_Applying", args.model)
    else:
        evaluate_scores(args.result_path, args.level, args.model)
