from data_augmentation import *
from moleDataset import *
import os
import json


augmention_path_location = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset_location.txt')
augmentation_path = None
with open(augmention_path_location, 'r') as file:
    augmentation_path = json.loads(file.read())
    augmentation_path = augmentation_path['aug_data']
    
    

dataset = MoleDataset()
augmenter = DataAugmenter(dataset=dataset, augmentation_path=augmentation_path)

# print(augmenter.data_distribution)
# print(augmenter.augmentation_needed)

augmenter.augment_multiple_classes(list(dataset.mapping.keys()), 2)
# augmenter.delete_augmented_data()
