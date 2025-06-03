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
import threading
from prompt_process import concept_prompt_process, task_prompt_process, integration_prompt_process

# 配置日志记录
log_filename = "./script_gpt4o_360_new.log"
logging.basicConfig(level=logging.ERROR, format="%(asctime)s - %(levelname)s - %(message)s", handlers=[logging.FileHandler(log_filename), logging.StreamHandler()])

class GPT4oService:
    def __init__(self, openai_ak=None):
        self.url = "https://api.openai.com/v1/chat/completions"
        self.headers = {"Authorization": f"Bearer {openai_ak}", "accept": "*/*", "Content-Type": "application/json"}

    def _gpt_response(self, user_prompt, image_generated, image_reference):
        api_input = {
            "model": "gpt-4o-0513-global", 
            "prompt": user_prompt,
            "stream": False,
            "temperature": 0.0,
        }

        # 构建请求的消息体
        messages = [
            {"role": "system", "content": "You are an image assessment expert."},
            {"role": "user", "content": [{"type": "text", "text": user_prompt}, {"type": "image_url", "image_url": {"url": image_generated, "detail": "auto"}}]},
        ]

        if image_reference:
            messages[1]["content"].append({"type": "image_url", "image_url": {"url": image_reference, "detail": "auto"}})

        api_input_final = {
            "messages": messages,
            "extendParams": {},
            "platformInput": api_input,
        }
        response = requests.post(self.url, headers=self.headers, data=json.dumps(api_input_final))
        # import pdb; pdb.set_trace()
        if response.status_code == 200:
            return json.loads(response.content.decode("utf-8"))
        else:
            logging.error(f"Request failed: {response.content}")
            return None

    def gpt_with_retry(self, user_prompt, image_generated, image_reference):
        retry = 5
        for _ in range(retry):
            try:
                result = self._gpt_response(user_prompt, image_generated, image_reference)
                if result is not None:
                    return result
            except Exception as e:
                print(f"An error occurred: {e}")
            time.sleep(1)
        return None

def process_data(data, gpt4o, image_base, level):
    if threading.current_thread().name != "MainThread":
        print(f"Running in thread: {threading.current_thread().name}")
    try:
        image_generated = data["result_image"].split("/")[-4] + "/" + data["result_image"].split("/")[-3] + "/" + data["result_image"].split("/")[-2] + "/" + data["result_image"].split("/")[-1]
        # print(image_generated)
        image_generated_path = os.path.join(image_base, image_generated)
        print(image_generated)
        text = data["sentence"]
        concepts = data["concept"]
        types = data["type"]
        reference_images = data["reference_image"]
        # dict_data = {"text": text, "concepts": concepts, "generated_image": data["result_image_best"]}
        concept_part_list = []
        # print(image_generated_path)
        false_flag = 0

        # concept Evaluation
        for concept, type_, reference_image in zip(concepts, types, reference_images):
            concept_image_path = os.path.join(image_base, "concept_image", reference_image.split("/")[-2], reference_image.split("/")[-1])
            print("concept_path:\n", concept_image_path)
            score_prompt_concept = concept_prompt_process(concept, type_, text)
            # print(score_prompt_individual)
            response = gpt4o.gpt_with_retry(score_prompt_concept, image_generated_path, concept_image_path)
            if response:
                if response["success"] == False:
                    res_concept = "None"
                    logging.error(f"概念：{concept_image_path}, 生成图: {image_generated_path}, {response}")
                    false_flag = 1
                else:
                    res_concept = response["data"]["choices"][0]["message"]["content"]
            else:
                res_concept = "None"
                false_flag = 1
            # print(response)
            print("Concept", concept)
            print("response:", res_concept)
            part_data = {"concept": concept, "concept_image": concept_image_path, "gpt4o_reasponse": res_concept}
            concept_part_list.append(part_data)

        if level == "Knowledge_Applying" or level == "Knowledge_Understanding":
            if level == "Knowledge_Applying" and len(data["concept"]) == 3:
                bandf = True
            else:
                bandf = False
            # print("Bandf:", bandf)

            # Task evaluation
            score_prompt_task = task_prompt_process(data, bandf)
            # print(score_prompt_integrated_eval)
            response = gpt4o.gpt_with_retry(score_prompt_task, image_generated_path, None)
            if response:
                if response["success"] == False:
                    res_task = "None"
                    logging.error(f"生成图Task: {image_generated_path}, {response}")
                    false_flag = 1
                else:
                    res_task = response["data"]["choices"][0]["message"]["content"]
            else:
                res_task = "None"
                false_flag = 1
            print(res_task)
            task_eva = res_task

            if level == "Knowledge_Applying":
                # print("Knowledge_Applying")  
                # integration evaluation        
                score_prompt_integration = integration_prompt_process(data)
                # print(score_prompt_integration)
                response = gpt4o.gpt_with_retry(score_prompt_integration, image_generated_path, None)
                # print("GPT4o_Response:", response)
                if response:
                    if response["success"] == False:
                        res_integration = "None"
                        logging.error(f"生成图Integration: {image_generated_path}, {response}")
                        false_flag = 1
                    else:
                        res_integration = response["data"]["choices"][0]["message"]["content"]
                else:
                    res_integration = "None"
                    false_flag = 1
                print(res_integration)
                integration_eva = res_integration
                data["concept_score"] = concept_part_list
                data["integration_score"] = integration_eva
                data["task_score"] = task_eva
                return data, false_flag
            else:
                data["concept_score"] = concept_part_list
                data["integration_score"] = ""
                data["task_score"] = task_eva
                return data, false_flag

        # "knowledge memorization"
        else:
            data["concept_score"] = concept_part_list
            data["integration_score"] = ""
            data["task_score"] = ""
            return data, false_flag
    except Exception as e:
        logging.error(f"Failed to process data: {data} - {e}")

        false_flag = 1
        # if false_flag == 1:
        return data, false_flag

def infer(result_path, model_name, level, eval_model):
    image_base = "https://mvap-public-data.oss-cn-zhangjiakou.aliyuncs.com/ziwei/T2I_Knowledge_Bench/"
    gpt4o = GPT4oService()
    print(result_path)
    # print(os.path.join(result_path, model_name,level))
    json_files = glob.glob(os.path.join(result_path, model_name,level, '*.jsonl'))
    print(json_files)
    for json_name in json_files:
        # if json_name == "/mnt/workspace/ziwei/MKC_results/ssr_encoder/level_three/level_three_livingsubject_common_livingsubject_common_landmark_common.json":
        #     continue

        # if json_name == "/mnt/workspace/ziwei/MKC_results/ssr_encoder/level_three/level_three_livingsubject_common_object_common_landmark_common.json":
        #     continue

        print(json_name)
        
        # if "differentiating" not in json_name:
        #     continue
        try:
            # 读取的json文件
            data_all = []
            with jsonlines.open(json_name, "r") as reader:
                for line in reader:
                    data_all.append(line)
        except Exception as e:
            logging.error(f"Failed to load JSON data: {e}")
            raise e
        results = []
        wrong_results = []
        
        # # temp
        # data_0 = []
        # for data in data_all:
        #     if data["type"] == ["event"]:
        #         data_0.append(data)

        wrong_results.extend(data_all)
        while wrong_results:
            with ThreadPoolExecutor(max_workers=10) as executor:
                futures = []
                for data in wrong_results:
                    # if data["result_image"] in remain_list:
                    future = executor.submit(process_data, data, gpt4o, image_base, level)

                    futures.append(future)
                    json_name_split = json_name.split("/")[-1].replace(".jsonl", "")
                for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing {json_name_split} images"):
                    try:
                        result, false_flag = future.result()
                        # print("false_flag", false_flag)
                        if false_flag == 0:
                            results.append(result)
                        else:
                            wrong_results.append(result)  
                    except Exception as e:
                        logging.error(f"Processing failed: {e}")

            results_texts = [item["sentence"] for item in results]
            wrong_results = [item for item in wrong_results if item["sentence"] not in results_texts]
            # print("results", results)
            # print("wrong_results", wrong_results)

        try:
            # 保存结果
            gpt4o_response_path = os.path.join(result_path, model_name ,level+f"_{eval_model}")
            os.makedirs(gpt4o_response_path, exist_ok=True)
            gpt4o_response_jsonl_path = os.path.join(gpt4o_response_path, json_name.split("/")[-1])
            with jsonlines.open(gpt4o_response_jsonl_path , "w") as writer:
                for result in results:
                    writer.write(result)
                    writer._fp.flush()
                print(f"{gpt4o_response_jsonl_path} writing finish")
        except Exception as e:
            logging.error(f"Failed to write results to file: {e}")

    logging.shutdown()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation Script")
    parser.add_argument("--result_path", type=str, required=True, help="Path to the result directory.")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--level", type=str, required=True, help="Level name for logging purposes.")
    parser.add_argument("--eval_model", type=str, required=True, help="Name of Evaluation Model")
    args = parser.parse_args()
    
    models = ["dalle3"]
    for model in models:
        if args.level == "level_all":
            infer(args.result_path, args.model_name, "Knowledge_Momerization", args.eval_model)
            infer(args.result_path, args.model_name, "Knowledge_Understanding", args.eval_model)
            infer(args.result_path, args.model_name, "Knowledge_Applying", args.eval_model)
            infer(args.result_path, args.model_name, "Knowledge_Memorization_add", args.eval_model)
        else:
            infer(args.result_path, model, args.level, args.eval_model)