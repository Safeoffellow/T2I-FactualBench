# import numpy as np
# from scipy import linalg
# import torch
# from torchvision import models, transforms
# from torch.nn.functional import adaptive_avg_pool2d
# from PIL import Image
# import os
# from typing import Literal, TypeAlias, List
# from tqdm.auto import tqdm

# def get_inception_model():
#     model = models.inception_v3(pretrained=True, transform_input=False)
#     model.eval()
#     return model

# def preprocess_image(image_path, transform):
#     image = Image.open(image_path).convert('RGB')
#     return image

# def get_features(model, images, device):
#     with torch.no_grad():
#         images = images.to(device)
#         preds = model(images)
#         print("***", preds)
#         preds = adaptive_avg_pool2d(preds, output_size=(1, 1))
#         preds = preds.squeeze(3).squeeze(2)
#     return preds.cpu().numpy()

# def calculate_fid(mu1, sigma1, mu2, sigma2, eps=1e-6):
#     diff = mu1 - mu2
#     covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
#     if np.iscomplexobj(covmean):
#         covmean = covmean.real
#     fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
#     return fid

# def compute_statistics(model, image_dir, transform, device):
#     features = []
#     for img_path in image_dir:
#         img = preprocess_image(img_path, transform)
#         feat = get_features(model, img, device)
#         features.append(feat)
#     features = np.concatenate(features, axis=0)
#     mu = np.mean(features, axis=0)
#     sigma = np.cov(features, rowvar=False)
#     return mu, sigma

# def calculate_fid_score(real_dir, generated_dir):
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model = get_inception_model().to(device)
#     transform = transforms.Compose([
#         transforms.Resize((299, 299)),
#         transforms.ToTensor(),
#     ])
#     mu1, sigma1 = compute_statistics(model, real_dir, transform, device)
#     mu2, sigma2 = compute_statistics(model, generated_dir, transform, device)
#     fid_value = calculate_fid(mu1, sigma1, mu2, sigma2)
#     return fid_value


# def single_gpu_eval_fid_score(image1_paths: List[str], image2_paths: List[str]) -> float:
#     assert len(image1_paths) == len(image2_paths), f"Number of image1 files {len(image1_paths)} != number of image2 files {len(image2_paths)}."

#     score = calculate_fid_score(image1_paths, image2_paths)
#     # print(score)
#     return score

import torch
import torchvision
import torchvision.transforms as transforms
from pytorch_fid import fid_score
from typing import Literal, TypeAlias, List

def single_gpu_eval_fid_score(image1_paths: List[str], image2_paths: List[str]) -> float:
    # 准备真实数据分布和生成模型的图像数据
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    assert len(image1_paths) == len(image2_paths), f"Number of image1 files {len(image1_paths)} != number of image2 files {len(image2_paths)}."

    # 加载预训练的Inception-v3模型
    inception_model = torchvision.models.inception_v3(pretrained=True)

    # 定义图像变换
    transform = transforms.Compose([
    transforms.Resize(299),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # 计算FID距离值
    print(image1_paths)
    paths = []
    for image1, image2 in zip(image1_paths, image2_paths):
        paths.append([image1, image2])
    fid_value = fid_score.calculate_fid_given_paths(paths, device=device, dims=2048, batch_size=50)
    print('FID value:', fid_value)
    
