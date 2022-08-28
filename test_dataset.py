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
                                        # transforms.ToTensor(),
                                        transforms.transforms.Normalize(mean, std),
                                        lambda image : transforms.functional.adjust_sharpness(image,
                                                                                              sharpness_factor=2)
])

dataset_pre = MoleDataset(transform=image_preprocessing)

dataset = MoleDataset()

i = 5000
cv2.imshow("Original", dataset[i][0])
cv2.imshow("Normalized and Sharpened", dataset[i][0])

cv2.waitKey(0)
    
# train_batch_size = 1
# train_loader = DataLoader(dataset=dataset_pre, batch_size=train_batch_size, shuffle=True)
# model = SkinCancerModel().to(device='cuda')

# for x, y, z in train_loader:
    # x = x.to(device='cuda')
    # print(model(x).shape)
    # break
    # print(y, "\n", x)
    # break

# print(np.unique(dataset.get_ground_truth(), return_counts=True))
