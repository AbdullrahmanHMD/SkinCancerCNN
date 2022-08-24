import os
from moleDataset import MoleDataset
import cv2
import json
import numpy as np

dataset = MoleDataset()

STATS_FILE_NAME = "dataset_stats.txt"
STATS_FILE_PATH = os.path.join(os.getcwd(), STATS_FILE_NAME)

mean, std = None, None

with open(STATS_FILE_PATH, 'r') as file:
    stats = json.loads(file.read())
    mean, std = stats['mean'], stats['std']


labels, mapped, mapping = dataset.get_ground_truth()

print(mapped[:10])
print(labels[:10])
print(mapping)

# print(np.unique(dataset.get_ground_truth(), return_counts=True))
