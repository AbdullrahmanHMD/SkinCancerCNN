from data_augmentation import *
from moleDataset import *
import os



# path = os.getcwd()

dataset = MoleDataset()
augmenter = DataAugmenter(dataset=dataset)

# print(augmenter.data_distribution)
# print(augmenter.augmentation_needed)

augmenter.augment_a_class('bkl', 10)
