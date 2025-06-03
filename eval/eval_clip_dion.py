import os
import json
import jsonlines
import glob
import argparse
import logging
from tool.dino import single_gpu_eval_dino_score, Dinov2Score
from tool.clip import single_gpu_eval_clipt_score, single_gpu_eval_clipi_score, CLIPScore
from datetime import datetime


# 设置日志记录
def setup_logging(level: str, model: str, clipt: str, text_knowledge: str):
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    if clipt == "all":
        if text_knowledge == "yes":
            log_dir = f"../log/{model}"
        else:
            log_dir = f"../log_text_knowledge_without/{model}"

    elif clipt == "only":
        log_dir = f"../log_without_ref/{model}"

    # 如果日志目录不存在，则创建
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    log_filename = f"{log_dir}/{level}_{current_time}_evaluation.log"
    logging.basicConfig(filename=log_filename, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def evaluate_scores(result_path, level, model, clipt, text_knowledge):
    setup_logging(level, model, clipt, text_knowledge)
    logging.info(f"Model_name: {model}")
    logging.info(f"Starting evaluation for level: {level}")

    result_path_json = os.path.join(result_path, model, level)
    json_files = glob.glob(os.path.join(result_path_json, "*.jsonl"))
    print(result_path_json)
    clipt = str(clipt)
    score_all_common = []
    score_all_uncommon = []

    score_all_clipt = []
    score_all_clipi = []
    score_all_dino = []
    for json_file in json_files:
        print(json_file)
        # if "object" not in json_file:
        #     continue
        data_all = []
        with jsonlines.open(json_file, "r") as reader:
            for line in reader:
                data_all.append(line)

        result_images = []
        reference_images = []
        concepts = []
        texts = []

        for data in data_all:
            result_images.append(data["result_image"])
            reference_images.append(data["reference_image"])
            concepts.append(data["concept"])
            if text_knowledge == "yes":
                texts.append(data["sentence"])
            else:
                texts.append(data["sentence_text_knowledge"])

        json_name = json_file.split("/")[-1].replace(".jsonl", "")
        # Compute scores
        # print(json_name)
        # print("参考图个数:", len(reference_images))
        # print(reference_images)
        # print(clipt)
        if clipt == "all":
            dino_score = single_gpu_eval_dino_score(reference_images, result_images, Dinov2Score())
            clipt_score_value = single_gpu_eval_clipt_score(texts, result_images, CLIPScore())
            clipi_score_value = single_gpu_eval_clipi_score(reference_images, result_images, CLIPScore())
            # Log the scores
            logging.info(f"Level: {level}")
            logging.info(f"Processed file: {json_name}")
            
            logging.info(f"dino_score: {dino_score}")
            logging.info(f"clipt_score: {clipt_score_value}")
            logging.info(f"clipi_score: {clipi_score_value}")

            score_all_clipt.append(clipt_score_value)
            score_all_clipi.append(clipi_score_value)
            score_all_dino.append(dino_score)
        elif clipt == "only":
            # print("only")
            clipt_score_value, clipt_best_dict = single_gpu_eval_clipt_score(texts, result_images, CLIPScore())
            # print(clipt_best_dict)
            logging.info(f"Level: {level}")
            logging.info(f"Processed file: {json_name}")
            logging.info(f"clipt_score: {clipt_score_value}")
            if "uncommon" in json_name:
                score_all_uncommon.append(clipt_score_value)
            else:
                score_all_common.append(clipt_score_value)

        # for clipt_best_temp in clipt_best_dict:
        #     for item in data_all:
        #         if item["result_image"] == clipt_best_temp['result_image']:
        #             item['result_image_best'] = clipt_best_temp["best_index"]
        #             break
        # with open(json_file, "w", encoding = "utf-8") as writer:
        #     json.dump(data_all, writer, ensure_ascii=False, indent=4)
        # print("score_all_common:", score_all_common)
        # print("score_all_uncommon", score_all_uncommon)

    # logging.info(f"Processed common_score_total: {sum(score_all_common)/len(score_all_common)}")
    # logging.info(f"Processed uncommon_score_total: {sum(score_all_uncommon)/len(score_all_uncommon)}\n")
    logging.info(f"Processed score_clipt: {sum(score_all_clipt)/len(score_all_clipt)}")
    logging.info(f"Processed score_clipi: {sum(score_all_clipi)/len(score_all_clipi)}")
    logging.info(f"Processed score_dino: {sum(score_all_dino)/len(score_all_dino)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation Script")
    parser.add_argument("--result_path", type=str, required=True, help="Path to the result directory.")
    parser.add_argument("--level", type=str, required=True, help="Level name for logging purposes.")
    parser.add_argument("--model", type=str, required=True, help="Model to generate the image.")
    parser.add_argument("--clipt", type=str, required=True, help="If need more metrics or not")
    parser.add_argument("--text_knowledge", type=str, required=True, help="If need text_knowledge")
    args = parser.parse_args()
    print("Current working directory:", os.getcwd())
    
    # model_list = ["fluxdev", "msdiffusion", "pixart", "playground", "sd1.5_new", "sd3.5_new", "sdXL", "ssr_encoder", "dalle3"]
    model_list = ["sd3.5_new_text_knowledge", "fluxdev_text_knowledge"]
    for model_name in model_list:
        if args.level == "level_all":
            evaluate_scores(args.result_path, "Knowledge_Momerization", model_name, args.clipt, args.text_knowledge)
            evaluate_scores(args.result_path, "Knowledge_Understanding", model_name, args.clipt, args.text_knowledge)
            evaluate_scores(args.result_path, "Knowledge_Applying", model_name, args.clipt, args.text_knowledge)
            evaluate_scores(args.result_path, "Knowledge_Memorization_add", model_name, args.clipt, args.text_knowledge)
        else:
            evaluate_scores(args.result_path, args.level, args.model, args.clipt, args.text_knowledge)
    
