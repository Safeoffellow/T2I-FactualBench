import math
import os
import torch
from accelerate import PartialState
import PIL.Image
import numpy as np
from tqdm.auto import tqdm
from transformers import CLIPModel, CLIPProcessor
import glob
from typing import Literal, TypeAlias, List
import json
from tool.comm import all_gather
import spacy
# import clip
nlp=spacy.load('en_core_web_sm')

# 定义常量
_DEFAULT_MODEL = "/mnt/workspace/ziwei/checkpoints/clip-vit-base-patch32"
_DEFAULT_TORCH_DTYPE: torch.dtype = torch.float32
IMAGE_EXT_LOWER = ["png", "jpeg", "jpg"]
IMAGE_EXT = IMAGE_EXT_LOWER + [_ext.upper() for _ext in IMAGE_EXT_LOWER]

"""
- pil: `PIL.Image.Image`, size (w, h), seamless conversion between `uint8`
- np: `np.ndarray`, shape (h, w, c), default `np.uint8`
- pt: `torch.Tensor`, shape (c, h, w), default `torch.uint8`
"""
ImageType: TypeAlias = PIL.Image.Image | np.ndarray | torch.Tensor
ImageTypeStr: TypeAlias = Literal["pil", "np", "pt"]
ImageFormat: TypeAlias = Literal["JPEG", "PNG"]
DataFormat: TypeAlias = Literal["255", "01", "11"]


def load_image(image_path: str, output_type: ImageTypeStr = "pil") -> ImageType:
    image = PIL.Image.open(image_path)
    if output_type == "pil":
        return image
    elif output_type == "np":
        return np.array(image)
    elif output_type == "pt":
        return torch.tensor(np.array(image).transpose(2, 0, 1), dtype=torch.float32)
    else:
        raise ValueError(f"Unsupported output_type: {output_type}")


class CLIPScore:
    def __init__(
        self,
        model_or_name_path: str = _DEFAULT_MODEL,
        torch_dtype: torch.dtype = _DEFAULT_TORCH_DTYPE,
        local_files_only: bool = False,
        device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__()
        self.device = device
        self.dtype = torch_dtype
        self.model = CLIPModel.from_pretrained(model_or_name_path, torch_dtype=torch_dtype, local_files_only=False).to(device)
        self.model.eval()
        self.processor = CLIPProcessor.from_pretrained(model_or_name_path, local_files_only=False)

    @torch.no_grad()
    def get_text_features(self, text: str | List[str], *, norm: bool = False) -> torch.Tensor:
        if not isinstance(text, list):
            text = [text]
        inputs = self.processor(text=text, padding=True, return_tensors="pt")
        text_features = self.model.get_text_features(
            inputs["input_ids"].to(self.device),
            inputs["attention_mask"].to(self.device),
        )
        if norm:
            text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        return text_features

    @torch.no_grad()
    def get_image_features(self, image: ImageType | List[ImageType], *, norm: bool = False) -> torch.Tensor:
        if not isinstance(image, list):
            image = [image]
        inputs = self.processor(images=image, return_tensors="pt")
        image_features = self.model.get_image_features(inputs["pixel_values"].to(self.device, dtype=self.dtype))
        if norm:
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        return image_features

    @torch.no_grad()
    def clipi_score(self, images1: ImageType | List[ImageType], images2: ImageType | List[ImageType]) -> tuple[float, int]:
        if not isinstance(images1, list):
            images1 = [images1]
        if not isinstance(images2, list):
            images2 = [images2]
        # print("image1_num", len(images1))
        assert len(images1) == len(images2) or len(images2) == 1, f"Number of images1 ({len(images1)}) and images2 ({len(images2)}) should be same."
        images2_features = self.get_image_features(images2, norm=True)
        score = 0
        if len(images1) > 1:
            for img in images1:
                images1_features = self.get_image_features(img, norm=True)
                # cosine similarity between feature vectors
                score += 100 * (images1_features * images2_features).sum(axis=-1)
                # print(score)
                print("score:", score.sum(0))
            return score.sum(0).float() / len(images1), len(images1)
        else:
            images1_features = self.get_image_features(images1, norm=True)
            # cosine similarity between feature vectors
            score = 100 * (images1_features * images2_features).sum(axis=-1)
            print("score:", score.sum(0).float())
            return score.sum(0).float(), len(images1)

    @torch.no_grad()
    def clipt_score(self, texts: str | List[str], images: ImageType | List[ImageType]) -> tuple[float, int]:
        if not isinstance(texts, list):
            texts = [texts]
        if not isinstance(images, list):
            images = [images]
        assert len(texts) == len(images), f"Number of texts ({len(texts)}) and images ({len(images)}) should be same."
        texts_new = []
        flag = 0
        for text in texts:
            if isinstance(text, str):
                flag = 1
                doc=nlp(text)
                prompt_without_adj=' '.join([token.text for token in doc if token.pos_ != 'ADJ']) #remove adj
                texts_new.append(prompt_without_adj)
                print(prompt_without_adj)
        if flag == 1:
            print("+++++NEW DINO+++++")
            texts_features = self.get_text_features(texts_new, norm=True)
        else:
            texts_features = self.get_text_features(texts, norm=True)
        images_features = self.get_image_features(images, norm=True)
        # cosine similarity between feature vectors
        score = 100 * (texts_features * images_features).sum(axis=-1)
        return score.sum(0).float(), len(texts)


def single_gpu_eval_clipi_score(image1_paths: List[str], image2_paths: List[str], clip_score: CLIPScore) -> float:
    if clip_score is None:
        clip_score = CLIPScore()
    assert len(image1_paths) == len(image2_paths), f"Number of image1 files {len(image1_paths)} != number of image2 files {len(image2_paths)}."

    total_score = 0.0
    pbar = tqdm(total=len(image1_paths), desc="Evaluating CLIP-I Score")

    for image1_path_list, image2_path in zip(image1_paths, image2_paths):
        print("个数:", len(image1_path_list))
        print("参考图:", image1_path_list)
        print("生成图：", image2_path)
        image2 = load_image(image2_path)
        image1 = []
        for image1_path in image1_path_list:
            # print("image1_path", image1_path)
            image1.append(load_image(image1_path))

        score, _ = clip_score.clipi_score(image1, image2)
        # print(score)
        total_score += score.item()
        pbar.update(1)
        # print("total_score:", total_score)

    pbar.close()
    return total_score / len(image1_paths)


def single_gpu_eval_clipt_score(texts: List[str], image_paths: List[str], clip_score: CLIPScore) -> float:
    if clip_score is None:
        clip_score = CLIPScore()
    
    image_num = 1
    assert len(texts) == len(image_paths), f"Number of texts ({len(texts)}) != number of image files {len(image_paths)}."
    print(texts)
    total_score = 0.0
    pbar = tqdm(total=len(texts), desc="Evaluating CLIP-T Score")
    clipt_best_dict = []
    for text, image_path in zip(texts, image_paths):
        print("*" * 10)
        print("text:", text)
        print("image_path", image_path)
        print("*" * 10)
        score_data = []
        # for index in range(image_num):
            # image_path_temp = image_path + f"_{index}.png"
        image_path_temp = image_path
        image = load_image(image_path_temp)
        score_temp, _ = clip_score.clipt_score(text, image)
        score_data.append(score_temp)

        score_ave = sum(score_data) / image_num
        score_best = max(score_data)
        score_best_index = score_data.index(score_best)
        dict_best = {"result_image": image_path, "best_index": image_path+"_"+str(score_best_index)+'.png', "best_score": score_best}
        clipt_best_dict.append(dict_best)
        total_score += score_ave.item()
        pbar.update(1)

    pbar.close()
    return total_score / len(texts)


def multi_gpu_eval_clipi_score(
    image1_paths: List[str],
    image2_paths: List[str],
    clip_score: CLIPScore | None = None,
    distributed_state: PartialState | None = None
) -> float:
    if distributed_state is None:
        distributed_state = PartialState()

    if clip_score is None:
        clip_score = CLIPScore(device=distributed_state.device)

    assert len(image1_paths) == len(image2_paths), f"Number of image1 files {len(image1_paths)} != number of image2 files {len(image2_paths)}."

    params = []
    for image1_path_list, image2_path in zip(image1_paths, image2_paths):
        params.append((image1_path_list, image2_path))

    pbar = tqdm(
        total=math.ceil(len(image1_paths) / distributed_state.num_processes),
        desc=f"Evaluating CLIP-I Score",
        disable=not distributed_state.is_local_main_process,
    )

    with distributed_state.split_between_processes(params) as sub_params:
        score = 0
        for _param in sub_params:
            image1_path_list, image2_path = _param
            # print("参考图个数: ", len(image1_path_list))
            # print("参考图：", image1_path_list)
            # print("结果图: ", image2_path)
            image2 = load_image(image2_path)

            image1 = []

            for image1_path in image1_path_list:
                image1.append(load_image(image1_path))

            score_temp, _ = clip_score.clipi_score(image1, image2)

            score += score_temp
            pbar.update(1)

    scores = all_gather(score)
    return (sum(scores) / len(image1_paths)).item()


def multi_gpu_eval_clipt_score(
    texts: List[str],
    image_paths: List[str],
    clip_score: CLIPScore | None = None,
    distributed_state: PartialState | None = None
) -> float:
    if distributed_state is None:
        distributed_state = PartialState()

    if clip_score is None:
        clip_score = CLIPScore(device=distributed_state.device)
    assert len(texts) == len(image_paths), f"Number of texts ({len(texts)}) != number of image files {len(image_paths)}."

    params = []
    for text, image_path in zip(texts, image_paths):
        params.append((text, image_path))
    pbar = tqdm(
        total=math.ceil(len(image_paths) / distributed_state.num_processes),
        desc=f"Evaluating CLIP-T Score",
        disable=not distributed_state.is_local_main_process,
    )

    with distributed_state.split_between_processes(params) as sub_params:
        score = 0
        for _param in sub_params:
            text, image_path = _param
            image = load_image(image_path)

            score_temp, _ = clip_score.clipt_score(text, image)

            score += score_temp
            pbar.update(1)

    scores = all_gather(score)
    return (sum(scores) / len(texts)).item()
