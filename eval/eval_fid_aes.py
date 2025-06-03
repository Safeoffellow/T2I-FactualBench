import os
import json
import jsonlines
import glob
import argparse
import logging
from tool.dino import single_gpu_eval_dino_score, Dinov2Score
from tool.clip import single_gpu_eval_clipt_score, single_gpu_eval_clipi_score, CLIPScore
from tool.fid_score import single_gpu_eval_fid_score
from tool_new.aes import single_gpu_eval_aes_score
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

    score_all_fid = []
    score_all_aes = []
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
            reference_images.append(data["reference_image"][0])
            concepts.append(data["concept"])
            if text_knowledge == "yes":
                texts.append(data["sentence"])
            else:
                texts.append(data["sentence_text_knowledge"])

        json_name = json_file.split("/")[-1].replace(".jsonl", "")

        fid_score = single_gpu_eval_fid_score(reference_images, result_images)
        aes_score = single_gpu_eval_aes_score(reference_images, result_images)
        print("Aeverage FID", fid_score)
        print("Aeverage AES", aes_score)
        # Log the scores
        logging.info(f"Level: {level}")
        logging.info(f"Processed file: {json_name}")

        logging.info(f"fid_score: {fid_score}")
        logging.info(f"aes_score: {aes_score}")

        score_all_fid.append(fid_score)
        score_all_aes.append(aes_score)

    logging.info(f"Processed score_fid: {sum(score_all_fid)/len(score_all_fid)}")
    logging.info(f"Processed score_aes: {sum(score_all_aes)/len(score_all_aes)}")

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
    model_list = ["fluxdev", "msdiffusion", "pixart", "playground", "sd1.5_new", "sd3.5_new", "sdXL", "ssr_encoder", "dalle3", "sd3.5_new_text_knowledge", "fluxdev_text_knowledge"]
    for model_name in model_list:
        if args.level == "level_all":
            evaluate_scores(args.result_path, "Knowledge_Momerization", model_name, args.clipt, args.text_knowledge)
            evaluate_scores(args.result_path, "Knowledge_Understanding", model_name, args.clipt, args.text_knowledge)
            evaluate_scores(args.result_path, "Knowledge_Applying", model_name, args.clipt, args.text_knowledge)
            evaluate_scores(args.result_path, "Knowledge_Memorization_add", model_name, args.clipt, args.text_knowledge)
        else:
            evaluate_scores(args.result_path, args.level, model_name, args.clipt, args.text_knowledge)
    
