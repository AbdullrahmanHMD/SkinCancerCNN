from genericpath import isdir
import os
from torch.utils.data import Dataset
import numpy as np
import os
import torch
import cv2
from torchvision import transforms as T
import pandas as pd

AUGMENTATION_PREFIX = 'AUG{}'
AUGMENTED_DATA_FILE_NAME = 'augmented_data'





DEFAULT_AUGMENTATION = T.Compose([
                                    T.ToTensor(),
                                    T.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
                                    T.GaussianBlur(kernel_size=3, sigma=(0.1, 5)),
                                    T.RandomVerticalFlip(p=0.75),
                                    T.RandomAutocontrast(p=0.5),
                                    
                                    # This transform makes the image readable by cv2.
                                    lambda image : torch.permute(image, [1, 2, 0])
                                  ])

class DataAugmenter():
    def __init__(self, dataset : Dataset, augmentation_transform=None, augmentation_path=None, max_points_per_class=3000):
        if augmentation_path is None:
            self.augmentation_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), AUGMENTED_DATA_FILE_NAME)
            if not isdir(self.augmentation_path):
                os.mkdir(self.augmentation_path)
        
        self.augmentation_transform = augmentation_transform
        
        if augmentation_transform is None:
            self.augmentation_transform = DEFAULT_AUGMENTATION
        
        
        
        self.dataset = dataset
        self.images_paths = dataset.images_paths
        self.data_distribution = dataset.get_data_distribution()
        self.augmentation_needed = self.get_augmentation_needed(max_points_per_class=max_points_per_class)
    
    def get_augmentation_needed(self, max_points_per_class=3000):
        """ Finds how many data points to augment per class.

        Args:
            max_points_per_class (int, optional): the max number
                of data points to be augmented.

        Returns:
            dict: A dictionary with keys being class names and values
                being how many data points are to be augmented for
                this class.
        """
        
        augmentation_num_per_class = {}
        for class_name, point_count in zip(self.data_distribution.keys(), self.data_distribution.values()):
            augmentation_num_per_class[class_name] = max_points_per_class - point_count
                   
        return augmentation_num_per_class
    
    def augment_a_class(self, class_label : str, aug_amount : int):
        
        class_indices = self.dataset.get_class_indecies(self.dataset.mapping[class_label])
        class_images_path = np.array(self.images_paths)[class_indices]
        
        images_to_augment = np.random.choice(class_images_path, aug_amount, replace=True)
        
        
        for image_path in images_to_augment:
            i = len(os.listdir(self.augmentation_path))
            
            # Readeing the image to be augmented:
            image = cv2.imread(image_path)
            
            # Converting the augmented image to a numpy array:
            augmented_image = self.augmentation_transform(image).cpu().detach().numpy()
            
            # Creating the path and name of the augmented image:
            # The naming convention is as follows:
            #   [the id of the original image] + _AUG + the number of augmentation,
            #   namely the variable i in this loop.
            image_path = image_path.split('.')[0]
            original_image_name = f'{AUGMENTATION_PREFIX.format(i)}_{os.path.basename(image_path)}.jpg'
            
            aug_image_path = os.path.join(self.augmentation_path, original_image_name)
            
            # Mutliplying the pixels by 255 to get the correct range:
            augmented_image = cv2.convertScaleAbs(augmented_image, alpha=(255.0))
            
            # Saving the augmented images into the given path:
            cv2.imwrite(aug_image_path, augmented_image)

            # cv2.imshow('augemnted', augmented_image)
            # cv2.imshow('original', image)
            # cv2.waitKey(0)
            
            
            
        
        
        