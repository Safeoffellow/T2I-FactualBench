import math
import os

# import fire
# import megfile
import torch
from accelerate import PartialState
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import BitImageProcessor, Dinov2Model
import PIL.Image
import numpy as np
import glob
from typing import Literal, List, TypeAlias
import json

# from typing import Literal,

# from dreambench_plus.constants import LOCAL_FILES_ONLY, MODEL_ZOOS
from tool.comm import all_gather
# from dreambench_plus.utils.image_utils import IMAGE_EXT, ImageType, load_image
# from dreambench_plus.utils.loguru import logger

# _DEFAULT_MODEL_V1: str = "dino_vits8"
# _DEFAULT_MODEL_V2: str = MODEL_ZOOS["facebook/dinov2-small"]
_DEFAULT_MODEL_V2 = "/mnt/workspace/ziwei/checkpoints/dinov2-small"
_DEFAULT_TORCH_DTYPE: torch.dtype = torch.float32

ImageType: TypeAlias = PIL.Image.Image | np.ndarray | torch.Tensor
ImageTypeStr: TypeAlias = Literal["pil", "np", "pt"]
ImageFormat: TypeAlias = Literal["JPEG", "PNG"]
DataFormat: TypeAlias = Literal["255", "01", "11"]


class Dinov2Score:
    # NOTE: noqa, in version 1, the performance of the official repository and HuggingFace is inconsistent.
    def __init__(
        self,
        model_or_name_path: str = _DEFAULT_MODEL_V2,
        torch_dtype: torch.dtype = _DEFAULT_TORCH_DTYPE,
        local_files_only: bool = False,
        device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        super().__init__()
        self.device = device
        self.dtype = torch_dtype
        self.model = Dinov2Model.from_pretrained(model_or_name_path, torch_dtype=torch_dtype, local_files_only=local_files_only).to(device)
        self.model.eval()
        self.processor = BitImageProcessor.from_pretrained(model_or_name_path, local_files_only=local_files_only)

    @torch.no_grad()
    def get_image_features(self, image: ImageType | list[ImageType], *, norm: bool = False) -> torch.Tensor:
        if not isinstance(image, list):
            image = [image]
        inputs = self.processor(images=image, return_tensors="pt")
        image_features = self.model(inputs["pixel_values"].to(self.device, dtype=self.dtype)).last_hidden_state[:, 0, :]
        if norm:
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        return image_features

    @torch.no_grad()
    def dino_score(self, images1: ImageType | list[ImageType], images2: ImageType | list[ImageType]) -> tuple[float, int]:
        if not isinstance(images1, list):
            images1 = [images1]
        if not isinstance(images2, list):
            images2 = [images2]
        assert len(images1) == len(images2) or len(images2) == 1, f"Number of images1 ({len(images1)}) and images2 ({len(images2)}) should be same."

        images2_features = self.get_image_features(images2, norm=True)
        # cosine similarity between feature vectors
        score = 0

        if len(images1) > 1:
            for img in images1:
                images1_features = self.get_image_features(img, norm=True)
                # cosine similarity between feature vectors
                score += 100 * (images1_features * images2_features).sum(axis=-1)
                print(score)
                print("score:", score.sum(0))
            return score.sum(0).float() / len(images1), len(images1)
        else:
            images1_features = self.get_image_features(images1, norm=True)
            # cosine similarity between feature vectors
            score = 100 * (images1_features * images2_features).sum(axis=-1)
            print("分数：", score.sum(0).float())
            return score.sum(0).float(), len(images1)


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


def single_gpu_eval_dino_score(
    image1_paths: List[str],
    image2_paths: List[str],
    dino_score: Dinov2Score | None = None,
    # version: Literal["v1", "v2"] = "v1",
) -> float:
    # if distributed_state is None:
    #     distributed_state = PartialState()

    # if dino_score is None:
    #     if version == "v1":
    #         dino_score = DinoScore(device=distributed_state.device)
    #     elif version == "v2":
    #         dino_score = Dinov2Score(device=distributed_state.device)
    #     else:
    #         raise ValueError(f"Invalid version {version}")
    if dino_score is None:
        dino_score = Dinov2Score()

    assert len(image1_paths) == len(image2_paths), f"Number of image1 files {len(image1_paths)} != number of image2 files {len(image2_paths)}."

    total_score = 0.0
    pbar = tqdm(total=len(image1_paths), desc="Evaluating DINO Score")

    for image1_path_list, image2_path in zip(image1_paths, image2_paths):
        print("参考图个数: ", len(image1_path_list))
        print("参考图：", image1_path_list)
        print("结果图: ", image2_path)
        image2 = load_image(image2_path)
        image1 = []
        for image1_path in image1_path_list:
            # print("image1_path", image1_path)
            image1.append(load_image(image1_path))

        score, _ = dino_score.dino_score(image1, image2)
        # print(score)
        total_score += score
        pbar.update(1)
        # print("total_score:", total_score)

    pbar.close()
    return total_score / len(image1_paths)


def multi_gpu_eval_dino_score(
    image1_paths: str,
    image2_paths: str,
    dino_score: Dinov2Score | None = None,
    distributed_state: PartialState | None = None,
    # version: Literal["v1", "v2"] = "v1",
) -> float:
    if distributed_state is None:
        distributed_state = PartialState()

    dino_score = Dinov2Score(device=distributed_state.device)

    assert len(image1_paths) == len(image2_paths), f"Number of image1 files {len(image1_files)} != number of image2 files {len(image2_files)}."

    params = []
    for image1_path_list, image2_path in zip(image1_paths, image2_paths):
        params.append((image1_path_list, image2_path))

    pbar = tqdm(
        total=math.ceil(len(image1_paths) / distributed_state.num_processes),
        desc=f"Evaluating Dino2 Score",
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

            score_temp, _ = dino_score.dino_score(image1, image2)

            score += score_temp
            pbar.update(1)

    scores = all_gather(score)
    return (sum(scores) / len(image1_paths)).item()
