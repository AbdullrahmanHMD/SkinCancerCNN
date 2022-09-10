import os
from torchvision import transforms
from data.moleDataset import MoleDataset
import numpy as np
import torch
import cv2
import json

from model import SkinCancerModel
from torch.utils.data import DataLoader


# --- Getting the mean and standard deviation of the data -----------------------------------------------
STATS_FILE_NAME = "dataset_stats.txt"
STATS_FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', STATS_FILE_NAME)


with open(STATS_FILE_PATH, 'r') as file:
    stats = json.loads(file.read())
    mean, std = stats['mean'], stats['std']
# -------------------------------------------------------------------------------------------------------

image_preprocessing = transforms.Compose([
                                        transforms.ToTensor(),
                                        transforms.transforms.Normalize(mean, std),
                                        lambda image : transforms.functional.adjust_sharpness(image,
                                                                                              sharpness_factor=2)
])

indecies = [0, 1, 2, 3]

# dataset_pre = MoleDataset(transform=image_preprocessing, indecies=indecies, augment=True)

# dataset_pre = MoleDataset(indecies=indecies, augment=True)

# for i in range(len(dataset_pre)):
#     image = dataset_pre[i][0]
#     cv2.imshow("Original", image)
#     cv2.waitKey(0)
    

import time

tic = time.time()
dataset_pre = MoleDataset(transform=transforms)
toc = time.time()

# print(f'Exec time: {toc - tic}')
    
