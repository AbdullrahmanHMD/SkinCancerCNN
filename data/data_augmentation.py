from genericpath import isdir
import os
from torch.utils.data import Dataset
import numpy as np
import os
import torch
import cv2
from torchvision import transforms as T
import pandas as pd
import re


# Defining constants:
#   1) AUGMENTAION_PREFIX is a prefix that is added
#       in the name of every augmented image.
AUGMENTATION_PREFIX = 'AUG{}'
#
#   2) The file name where the augmented images and
#       their metadata are to be located.
AUGMENTED_DATA_FILE_NAME = 'augmented_data'
#
#   3) The file name of the csv file that contains
#       the metadata of the dataset.
METADATA_FILE_NAME = 'metadata.csv'
#
#   4) The defualt augmentation transform which is
#       applied if no transform is provided.
DEFAULT_AUGMENTATION = T.Compose([
                                    
                                    # T.ToTensor(),
                                    T.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
                                    T.GaussianBlur(kernel_size=3, sigma=(0.1, 1)),
                                    T.RandomVerticalFlip(p=0.75),
                                    T.RandomAutocontrast(p=0.5),
                                    lambda image : torch.permute(image, [1, 2, 0])
                                    # This transform makes the image readable by cv2.
                                    
                                  ])

class DataAugmenter():
    def __init__(self, dataset : Dataset, augmentation_transform=None, augmentation_path=None, max_points_per_class=3000):
        self.augmentation_path = augmentation_path
        
        if augmentation_path is None:
            self.augmentation_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), AUGMENTED_DATA_FILE_NAME)
        
        # If the given augmentation path does not exist, create a folder
        # corresponding to the given path
        if not isdir(self.augmentation_path):
            os.mkdir(self.augmentation_path)
        
        self.augmentation_transform = augmentation_transform
        
        if augmentation_transform is None:
            self.augmentation_transform = DEFAULT_AUGMENTATION
        
        self.metadata_columns = ['lesion_id', 'image_id', 'dx', 'dx_type', 'age', 'sex', 'localization']
        
        aug_metadata_path = os.path.join(self.augmentation_path, 'metadata.csv')
        
        if not os.path.exists(aug_metadata_path):
            self.df = pd.DataFrame(columns=self.metadata_columns)
            self.df.to_csv(aug_metadata_path, index=False)
        
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
        """ Creates augmented images given a class. Takes a random sample
        from the given class and creates aug_amount amount of augmented
        images. Finally, saves the augmented images along with their
        metadata to the path provided to the DataAugmenter instance.

        Args:
            class_label (str): The class to augment images for.
            aug_amount (int): The number of data points to augment.
        """
        # Getting the indicies of the data points of the given class:
        class_indecies = self.dataset.get_class_indecies(self.dataset.mapping[class_label])
        # Taking a random sample from the indecies and images corresponding to
        # those indecies:
        aug_indicies = np.random.choice(class_indecies, aug_amount, replace=True)
        for image_index in aug_indicies:
            i = len(os.listdir(self.augmentation_path))
            
            # -----------------------------------------------------------------------------------
            # --- Reading and augmenting the images: --------------------------------------------
            
            image_path = self.images_paths[image_index]
            
            # Reading the image to be augmented:
            image = self.dataset[image_index][0]
            
            # Applying the augmentation:
            augmented_image = self.augmentation_transform(image).cpu().detach().numpy()
            
            # Creating the path and name of the augmented image:
            # The naming convention is as follows:
            #   [the id of the original image] + _AUG + the number of augmentation,
            #   namely the variable i in this loop.
            image_path = image_path.split('.')[0]
            aug_image_name = f'{AUGMENTATION_PREFIX.format(i)}_{os.path.basename(image_path)}'
            original_image_name = f'{aug_image_name}.jpg'
            
            aug_image_path = os.path.join(self.augmentation_path, original_image_name)
            
            # Mutliplying the pixels by 255 to get the correct range:
            augmented_image = cv2.convertScaleAbs(augmented_image, alpha=(255.0))
            
            # Saving the augmented images into the given path:
            cv2.imwrite(aug_image_path, augmented_image)

            # -----------------------------------------------------------------------------------
            # --- Providing the label and metadata: --------------------------------------------
            
            # Getting the metadata of the corresponding image:
            metadata = self.dataset[image_index][2]
            # Updating the image_id with the augmented image name:
            metadata['image_id'] = aug_image_name
            
            columns = metadata.to_numpy()
            columns[0], columns[1] = columns[1], columns[0]
            
            temp_dict = dict(zip(columns, metadata))
            
            # The path where the augmented images will be saved:
            metadata_path = os.path.join(self.augmentation_path, METADATA_FILE_NAME)
            
            metadata_df = pd.DataFrame()
            metadata_df = metadata_df.append(temp_dict, ignore_index=True)
            
            # Adding the metadata to the metadata.csv file:
            metadata_df.to_csv(metadata_path, mode='a', index=False, header=False)
            
            # cv2.imshow(f'augemnted: {label}', augmented_image)
            # cv2.imshow(f'original: {label}', image)
            # cv2.waitKey(0)
            
    def augment_multiple_classes(self, classes : list, aug_amount, replace=False):
        
        # If replace is true, all previously augmented data will be
        # deleted.
        if replace:
            self.delete_augmented_data()
            self.df = pd.DataFrame(columns=self.metadata_columns)
            self.df.to_csv(os.path.join(self.augmentation_path, 'metadata.csv'), index=False)
                        
        if isinstance(aug_amount, int):
            aug_amount = [aug_amount] * len(classes)
            
        for cls, amount in zip(classes, aug_amount):
            print(f'class: {cls} | amount augmented: {amount}')
            self.augment_a_class(cls, amount)
            
    def delete_augmented_data(self):
        """ Deletes the augmented data points in the self.augmentation_path
            path along with the metadata.csv file.
        """
        safe_to_delete = False
        
        images_num = 0
        csv_num = 0
        other = 0
        
        # Checks if it is "safe" to delete all the augmented images:
        # This check consists of checking that all the images in the
        # folder are of the form AUG[0-9]+_ISIC_[0-9]+. and there is
        # a metadata.csv file and there is no other file.
        for file_name in os.listdir(self.augmentation_path):
            if re.search('AUG[0-9]+_ISIC_[0-9]+.', file_name):
                images_num += 1
            elif re.search('metadata.csv', file_name):
                csv_num += 1
            else:
                other += 1
        
        safe_to_delete = images_num and csv_num and not other
        
        # Delete everything if it is safe to delete:
        if safe_to_delete:
            for file_name in os.listdir(self.augmentation_path):
                image_to_remove_path = os.path.join(self.augmentation_path, file_name)
                os.remove(image_to_remove_path)
        # Show an error message otherwise:
        else:
            error_message = """Cannot delete the augmented images due to at least one of the following reasons:
1) The augemented images' folder contains a file that is not an augmented image or the metadata csv file.
2) The augemented images' folder does not contain any augmented image.
3) The augmented images' folder does not contain the metadatacsv file
"""   
            print(error_message)

            
    
    
            
        
        
        