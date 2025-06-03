# import os, sys, time, json
# root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# print("root_dir:{}".format(root_dir))
# sys.path.append(root_dir)

# import torch
# import common_io
# from external.open_muse.muse import MLP
# import os
# import time
# import numpy as np
# # os.environ["CUDA_VISIBLE_DEVICES"] = "0"    # choose GPU if you are on a multi GPU server
# import torch
# import tqdm
# import json
# import clip
# from torch.utils.data import DataLoader
# import torch
# import time
# import traceback
# from tqdm import tqdm
# # init model
# model1_path = "/mnt/workspace/ziwei/checkpoints/aesthetic-predictor/sac+logos+ava1-l14-linearMSE.pth"
# model2_path = "/mnt/workspace/ziwei/checkpoints/aesthetic-predictor/ViT-L-14.pt"
# # model_path = "/root/.cache/huggingface/open_muse/vqgan-f16-8192-laion/"
# # print("load model:{}".format(model_path))

# model = MLP(768)  # CLIP embedding dim is 768 for CLIP ViT L 14

# s = torch.load(model1_path)   # load the model you trained previously or the model available in this repo

# model.load_state_dict(s)

# model.to("cuda")
# model.eval()

# device = "cuda" if torch.cuda.is_available() else "cpu"
# print(device)

# model2, preprocess = clip.load(model2_path, device=device)  #RN50x64   
# model2.eval()

# with torch.no_grad():
#     image_features = model2.encode_image(image)
#     if debug:
#         print("image_features time:{}".format(time.time()-t1))
#         t2 = time.time()
#     # convert to model type
#     im_emb_arr = normalized_tensor(image_features).to(model.dtype)

#     if debug:
#         print("normalized_tensor time:{}".format(time.time()-t2))
#         t3 = time.time()


#     aesthetic_scores = model(im_emb_arr)
#     if debug:
#         print("model time:{}".format(time.time()-t3))
#         t4 = time.time()

# # print("aesthetic_scores:{}".format(aesthetic_scores.flatten().tolist()))
# aesthetic_scores = aesthetic_scores.flatten().tolist()
# # convert element of aesthetic_scores to string
# aesthetic_scores = [str(x) for x in aesthetic_scores]

# print("model loaded!")
# print("model device:{}".format(model.device))
# print("model dtype:{}".format(model.dtype))
# print("model2:{}".format(model2))

from PIL import Image
import io
import matplotlib.pyplot as plt
import os
import json

from warnings import filterwarnings
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"    # choose GPU if you are on a multi GPU server
import numpy as np
import torch
import pytorch_lightning as pl
import torch.nn as nn
from torchvision import datasets, transforms
import tqdm

from os.path import join
from datasets import load_dataset
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import json
import clip
from PIL import Image, ImageFile
#####  This script will predict the aesthetic score for this image file:
img_path = "./American_bison_Black_vulture_Kinderdijk_aggressive_fearful.png"
# if you changed the MLP architecture during training, change it also here:
class MLP(pl.LightningModule):
    def __init__(self, input_size, xcol='emb', ycol='avg_rating'):
        super().__init__()
        self.input_size = input_size
        self.xcol = xcol
        self.ycol = ycol
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            #nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            #nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            #nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(64, 16),
            #nn.ReLU(),

            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layers(x)

    def training_step(self, batch, batch_idx):
            x = batch[self.xcol]
            y = batch[self.ycol].reshape(-1, 1)
            x_hat = self.layers(x)
            loss = F.mse_loss(x_hat, y)
            return loss
    
    def validation_step(self, batch, batch_idx):
        x = batch[self.xcol]
        y = batch[self.ycol].reshape(-1, 1)
        x_hat = self.layers(x)
        loss = F.mse_loss(x_hat, y)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

def normalized(a, axis=-1, order=2):
    import numpy as np  # pylint: disable=import-outside-toplevel

    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)


model = MLP(768)  # CLIP embedding dim is 768 for CLIP ViT L 14

s = torch.load("/mnt/workspace/ziwei/checkpoints/aesthetic-predictor/sac+logos+ava1-l14-linearMSE.pth")   # load the model you trained previously or the model available in this repo

model.load_state_dict(s)

model.to("cuda")
model.eval()


device = "cuda" if torch.cuda.is_available() else "cpu"
model2, preprocess = clip.load("/mnt/workspace/ziwei/checkpoints/aesthetic-predictor/ViT-L-14.pt", device=device)  #RN50x64   

pil_image = Image.open(img_path)

image = preprocess(pil_image).unsqueeze(0).to(device)

with torch.no_grad():
   image_features = model2.encode_image(image)

im_emb_arr = normalized(image_features.cpu().detach().numpy() )

prediction = model(torch.from_numpy(im_emb_arr).to(device).type(torch.cuda.FloatTensor))

print( "Aesthetic score predicted by the model:")
print( prediction )