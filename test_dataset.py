import os
from torchvision import transforms
from data.moleDataset import MoleDataset
import numpy as np
import torch
import cv2
import json

from model import SkinCancerModel
from torch.utils.data import DataLoader


image_preprocessing = transforms.Compose([
                                        transforms.ToTensor(),
                                        # transforms.transforms.Normalize((0.1794, 0.1794, 0.1794), (0.1884, 0.1884, 0.1884)),
                                        # lambda image : transforms.functional.adjust_sharpness(image, sharpness_factor=2)
])

dataset = MoleDataset(transform=image_preprocessing)
dataset.get_ground_truth()
STATS_FILE_NAME = "dataset_stats.txt"
STATS_FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', STATS_FILE_NAME)

mean, std = None, None

with open(STATS_FILE_PATH, 'r') as file:
    stats = json.loads(file.read())
    mean, std = stats['mean'], stats['std']
    
train_batch_size = 1
train_loader = DataLoader(dataset=dataset, batch_size=train_batch_size, shuffle=True)
model = SkinCancerModel().to(device='cuda')

for x, y, z in train_loader:
    x = x.to(device='cuda')
    print(model(x).shape)
    break
    # print(y, "\n", x)
    # break

# print(np.unique(dataset.get_ground_truth(), return_counts=True))
