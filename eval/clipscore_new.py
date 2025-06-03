import io
import os
from typing import Literal, TypeAlias

import megfile
import numpy as np
import PIL.Image
import PIL.ImageOps
import requests
import torch
import functools
import torch.distributed as dist

# from .loguru import logger

"""
- pil: `PIL.Image.Image`, size (w, h), seamless conversion between `uint8`
- np: `np.ndarray`, shape (h, w, c), default `np.uint8`
- pt: `torch.Tensor`, shape (c, h, w), default `torch.uint8`
"""
ImageType: TypeAlias = PIL.Image.Image | np.ndarray | torch.Tensor
ImageTypeStr: TypeAlias = Literal["pil", "np", "pt"]
ImageFormat: TypeAlias = Literal["JPEG", "PNG"]
DataFormat: TypeAlias = Literal["255", "01", "11"]


IMAGE_EXT_LOWER = ["png", "jpeg", "jpg"]
IMAGE_EXT = IMAGE_EXT_LOWER + [_ext.upper() for _ext in IMAGE_EXT_LOWER]


def check_image_type(image: ImageType):
    if not (isinstance(image, PIL.Image.Image) or isinstance(image, np.ndarray) or isinstance(image, torch.Tensor)):
        raise TypeError(f"`image` should be PIL Image, ndarray or Tensor. Got `{type(image)}`.")


def load_image(
    image: str | os.PathLike | PIL.Image.Image,
    *,
    output_type: ImageTypeStr = "pil",
) -> ImageType:
    """
    Loads `image` to a PIL Image, NumPy array or PyTorch tensor.

    Args:
        image (str | PIL.Image.Image): The path to image or PIL Image.
        mode (ImageMode, optional): The mode to convert to. Defaults to None (no conversion).
            The current version supports all possible conversions between "L", "RGB", "RGBA".
        output_type (ImageTypeStr, optional): The type of the output image. Defaults to "pil".
            The current version supports "pil", "np", "pt".

    Returns:
        ImageType: The loaded image in the given type.
    """
    timeout = 10
    # Load the `image` into a PIL Image.
    if isinstance(image, str) or isinstance(image, os.PathLike):
        if image.startswith("http://") or image.startswith("https://"):
            try:
                image = PIL.Image.open(requests.get(image, stream=True, timeout=timeout).raw)
            except requests.exceptions.Timeout:
                raise ValueError(f"HTTP request timed out after {timeout} seconds")
        elif image.startswith("s3"):
            with megfile.smart_open(image, "rb") as f:
                bytes_data = f.read()
            image = PIL.Image.open(io.BytesIO(bytes_data), "r")
        elif os.path.isfile(image):
            image = PIL.Image.open(image)
        else:
            raise ValueError(f"Incorrect path or url, URLs must start with `http://`, `https://` or `s3+[profile]://`, and `{image}` is not a valid path.")
    elif isinstance(image, PIL.Image.Image):
        image = image
    else:
        raise ValueError(f"`image` must be a path or PIL Image, got `{type(image)}`")

    # Automatically adjust the orientation of the image to match the direction it was taken.
    image = PIL.ImageOps.exif_transpose(image)

    support_mode = ["L", "RGB", "RGBA", "CMYK"]
    if image.mode not in support_mode:
        raise ValueError(f"Only support mode in `{support_mode}`, got `{image.mode}`")

    # add white background for RGBA images, and convert to RGB
    if image.mode == "RGBA":
        background = PIL.Image.new("RGBA", image.size, "white")
        image = PIL.Image.alpha_composite(background, image).convert("RGB")

    image = image.convert("RGB")

    if output_type == "pil":
        image = image
    elif output_type == "np":
        image = to_np(image)
    elif output_type == "pt":
        image = to_pt(image)
    else:
        raise ValueError(f"`output_type` must be one of `{ImageTypeStr}`, got `{output_type}`")

    return image


@functools.lru_cache()
def _get_global_gloo_group():
    """
    Return a process group based on gloo backend, containing all the ranks
    The result is cached.
    """
    if dist.get_backend() == "nccl":
        return dist.new_group(backend="gloo")
    else:
        return dist.group.WORLD


def all_gather(data, group=None):
    """
    Run all_gather on arbitrary picklable data (not necessarily tensors).

    Args:
        data: any picklable object
        group: a torch process group. By default, will use a group which
            contains all ranks on gloo backend.

    Returns:
        list[data]: list of data gathered from each rank
    """
    if group is None:
        group = _get_global_gloo_group()  # use CPU group by default, to reduce GPU RAM usage.
    world_size = dist.get_world_size(group)
    if world_size == 1:
        return [data]

    device = data.device
    output = [None for _ in range(world_size)]
    dist.all_gather_object(output, data, group=group)
    output = [o.to(device) for o in output]
    return output


def initialize_distributed_backend(backend="nccl", init_method="env://"):
    """
    Initializes the default distributed process group.
    Args:
        backend (str): Backend to use (nccl, gloo).
        init_method (str): URL specifying how to initialize the process group.
    """
    dist.init_process_group(backend=backend, init_method=init_method)


import math
import os
from typing import Literal
import fire
import megfile
import torch
import PIL.Image
import numpy as np
from accelerate import PartialState
from tqdm.auto import tqdm
from transformers import CLIPModel, CLIPProcessor
import glob
from typing import Literal, TypeAlias
import json

# from dreambench_plus.constants import LOCAL_FILES_ONLY, MODEL_ZOOS
# from dreambench_plus.utils.comm import all_gather
# from dreambench_plus.utils.image_utils import IMAGE_EXT, ImageType, load_image
# from dreambench_plus.utils.loguru import logger

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
    def get_text_features(self, text: str | list[str], *, norm: bool = False) -> torch.Tensor:
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
    def get_image_features(self, image: ImageType | list[ImageType], *, norm: bool = False) -> torch.Tensor:
        if not isinstance(image, list):
            image = [image]
        inputs = self.processor(images=image, return_tensors="pt")
        image_features = self.model.get_image_features(inputs["pixel_values"].to(self.device, dtype=self.dtype))
        if norm:
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        return image_features

    @torch.no_grad()
    def clipi_score(self, images1: ImageType | list[ImageType], images2: ImageType | list[ImageType]) -> tuple[float, int]:
        if not isinstance(images1, list):
            images1 = [images1]
        if not isinstance(images2, list):
            images2 = [images2]
        assert len(images1) == len(images2), f"Number of images1 ({len(images1)}) and images2 {(len(images2))} should be same."

        images1_features = self.get_image_features(images1, norm=True)
        images2_features = self.get_image_features(images2, norm=True)
        # cosine similarity between feature vectors
        score = 100 * (images1_features * images2_features).sum(axis=-1)
        return score.sum(0).float(), len(images1)

    @torch.no_grad()
    def clipt_score(self, texts: str | list[str], images: ImageType | list[ImageType]) -> tuple[float, int]:
        if not isinstance(texts, list):
            texts = [texts]
        if not isinstance(images, list):
            images = [images]
        assert len(texts) == len(images), f"Number of texts ({len(texts)}) and images {(len(images))} should be same."

        texts_features = self.get_text_features(texts, norm=True)
        images_features = self.get_image_features(images, norm=True)
        # cosine similarity between feature vectors
        score = 100 * (texts_features * images_features).sum(axis=-1)
        return score.sum(0).float(), len(texts)


def multigpu_eval_clipi_score(
    image1_dir: list,
    image2_dir: list,
    distributed_state: PartialState | None = None,
    clip_score: CLIPScore | None = None,
) -> float:
    if distributed_state is None:
        distributed_state = PartialState()

    if clip_score is None:
        clip_score = CLIPScore(device=distributed_state.device)

    assert len(image1_dir) == len(image2_dir), f"Number of image1 files {len(image1_dir)} != number of image2 files {len(image2_dir)}."

    params = []
    for image1_file, image2_file in zip(image1_dir, image2_dir):
        # assert (
        #     image1_file.split(image1_dir)[-1].split(".")[0] == image2_file.split(image2_dir)[-1].split(".")[0]
        # ), f"Image1 file {image1_file} and image2 file {image2_file} do not match."

        params.append((image1_file, image2_file))

    pbar = tqdm(
        total=math.ceil(len(image1_dir) / distributed_state.num_processes),
        desc="Evaluating CLIP-I Score",
        disable=not distributed_state.is_local_main_process,
    )

    with distributed_state.split_between_processes(params) as sub_params:
        score = 0
        for _param in sub_params:
            image1_file, image2_file = _param
            image1, image2 = load_image(image1_file), load_image(image2_file)
            score += clip_score.clipi_score(image1, image2)[0]
            pbar.update(1)

    scores = all_gather(score)
    return (sum(scores) / len(image1_dir)).item()


def multigpu_eval_clipt_score(
    text_dir: list,
    image_dir: list,
    distributed_state: PartialState | None = None,
    clip_score: CLIPScore | None = None,
) -> float:
    if distributed_state is None:
        distributed_state = PartialState()

    if clip_score is None:
        clip_score = CLIPScore(device=distributed_state.device)

    assert len(text_dir) == len(image_dir), f"Number of text files {len(text_dir)} != number of image files {len(image_dir)}."

    params = []
    for text_file, image_file in zip(text_dir, image_dir):
        params.append((text_file, image_file))

    pbar = tqdm(
        total=math.ceil(len(text_dir) / distributed_state.num_processes),
        desc="Evaluating CLIP-T Score",
        disable=not distributed_state.is_local_main_process,
    )

    with distributed_state.split_between_processes(params) as sub_params:
        score = 0
        for _param in sub_params:
            text_file, image_file = _param
            image = load_image(image_file)
            score += clip_score.clipt_score(text_file, image)[0]
            pbar.update(1)

    scores = all_gather(score)
    return (sum(scores) / len(text_dir)).item()


def clip_eval(mode: Literal["clipi", "clipt"], dir1: str, dir2: str):
    if mode == "clipi":
        logger.info(f"CLIP-I Score: {multigpu_eval_clipi_score(dir1, dir2)}")
    elif mode == "clipt":
        logger.info(f"CLIP-T Score: {multigpu_eval_clipt_score(dir1, dir2)}")


if __name__ == "__main__":
    initialize_distributed_backend()
    result_path = "/mnt/workspace/ziwei/SSR_ENcoder/SSR_Encoder/results/KBR_bench/level_one/"

    json_files = glob.glob(os.path.join(result_path, "*.json"))

    for json_file in json_files:
        with open(json_file, "r") as reader:
            data_all = json.load(reader)
            result_images = []
            reference_images = []
            texts = []
            for data in data_all:
                result_images.append(data["result_image"])
                reference_images.append(data["reference_image"])
                texts.append(data["text"])
            clipt_score = multigpu_eval_clipt_score(texts, result_images)
            clipi_score = multigpu_eval_clipi_score(reference_images, result_images)
            print("clipt_score:", clipt_score)
            print("clipi_score:", clipi_score)
