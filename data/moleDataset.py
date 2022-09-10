from importlib.metadata import metadata
from torch.utils.data import Dataset
import os
import pandas as pd
import json
import cv2
import numpy as np
from numpy import copy



class MoleDataset(Dataset):
    """_summary_

    Args:
        Dataset (MoleDataset): _description_
    """
    def __init__(self, dataset_path=None, labels_path=None, transform=None, indecies=None, augment=False):
        self.dataset_path = dataset_path
        self.labels_path = labels_path
        self.transform = transform
        
        self.indecies = indecies
        self.augment = augment
        
        self.images_paths = self.read_data()
        self.metadata = self.load_metadata()
        self.aug_metadata = None
        self.labels, self.mapped_labels, self.mapping = self.get_ground_truth()
    
    
    def read_data(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        # If the dataset path is not provided then read from the dataset_location.txt
        # file, otherwise use the value provided when creating the MoleDataset object.
        if self.dataset_path is None:
            data_loc_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset_location.txt")
            
            with open(data_loc_path, 'r') as file:
                data_dict = json.loads(file.read())
                
            self.dataset_path = data_dict['data']
               
        # Given a list of data points, joins the name of each data point with the
        # path of the data.
        join_with_data_path = lambda x, abs_path : os.path.join(abs_path, x)
        
        images_paths = os.listdir(self.dataset_path)
        images_paths = list(map(join_with_data_path, images_paths, [self.dataset_path] * len(images_paths)))
        
        # If the indecies of the data is specified then select only the images
        # corresponding to the specified indecies.
        if self.indecies is not None:
            images_paths = np.array(images_paths)[self.indecies]
        
        # If augment is true, reads appends the paths of the augmented images to the
        # list of images' paths.
        if self.augment:
            with open(data_loc_path, 'r') as file:
                data_dict = json.loads(file.read())
            # Reading the path where the augmented data is located:
            aug_data_path = data_dict['aug_data']
            # Creating a list that contains the names of the augmented images:
            aug_images_paths = os.listdir(aug_data_path)
            aug_images_paths.remove('metadata.csv')
            # Joining the path of the folder that contains the augmented images
            # with the images' names:
            aug_images_paths = np.array(list(map(join_with_data_path, aug_images_paths, [aug_data_path] * len(aug_images_paths))))
            # Joining the original and augmented data paths together:
            images_paths = np.append(images_paths, aug_images_paths)
            
        return images_paths
    
    
    def load_metadata(self):
        """ This method assigns all the metadata provided in the dataset
            to its corresponding image.

        Returns:
            list: a list of labels that correspond to the images.
        """
        # If the labels' path is not provided then read from the dataset_location.txt
        # file, otherwise use the value provided when creating the MoleDataset object.
        if self.labels_path is None:
            data_loc_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset_location.txt")
            
            with open(data_loc_path, 'r') as file:
                data_dict = json.loads(file.read())
                
            self.labels_path = data_dict['labels']        
        
        metadata = pd.read_csv(self.labels_path)
        
        if self.augment:
            data_loc_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset_location.txt")
            
            with open(data_loc_path, 'r') as file:
                data_dict = json.loads(file.read())
            
            aug_metadata_path = data_dict['aug_data']
            aug_metadata_path = os.path.join(aug_metadata_path, 'metadata.csv')
            aug_metadata = pd.read_csv(aug_metadata_path)
            self.aug_metadata = aug_metadata
            
            metadata = pd.concat([metadata, aug_metadata])
            
        return metadata

    
    def __getitem__(self, index):
        
        # Getting the image path and reading it via cv2:
        image_path = self.images_paths[index]
        image = cv2.imread(image_path)
        
        if self.transform is not None:
            image = self.transform(image)
        
        # Getting the label and metadata of the given image:
        image_id = os.path.basename(os.path.normpath(image_path)).split('.')[0]
        COLUMN_NAME = 'image_id'
        metadata = self.metadata.loc[self.metadata[COLUMN_NAME] == image_id].to_dict('list')
        
        return image, self.mapped_labels[index], metadata
   
    
    def __len__(self):
        return len(self.images_paths)
    
    
    def get_ground_truth(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        COLUMN_NAME = 'image_id'

        # Getting the names of all images from the images' paths:
        image_ids = [os.path.basename(os.path.normpath(im)).split('.')[0].strip() for im in self.images_paths]
        # Getting the metadata from the wanted images (Wanted images are the images specified by the indecies
        # variable and/or the augmented data)
        image_metadata = self.metadata.loc[np.isin(self.metadata.to_numpy(), image_ids, assume_unique=True)].to_numpy()
        
        # Retrieving the labels from the metadata:
        labels = image_metadata[:, 2]
        
        unique_values = np.unique(labels)
        # # The mapping of the categorical values to numerical values:
        mapping = dict(zip(unique_values, list(range(len(unique_values)))))
        
        # # Creating the mapped labels:
        mapped_labels = copy(labels)
        for val in unique_values:
            feat_mapping = mapping[val]
            mapped_labels[labels==val] = feat_mapping
            
        return labels, mapped_labels.astype(np.int64), mapping
    
    
    def get_data_distribution(self):
        """ Provides a dictionary with keys being the classes names and values
            being the frequency of these classes in the dataset.

        Returns:
            dict: A dictionary with keys being the classes and values being the
            frequency of these classes.
        """
        data_distribution = dict(zip(self.mapping.keys(), np.bincount(self.mapped_labels)))
        return data_distribution
    
    
    def get_class_indecies(self, label : int):
        """ Gets the indicies of a data point of a certain class.

        Args:
            label (int): The label of the class in whose indicies
            will be returned.

        Returns:
            list: The list of the indicies of the given class.
        """
        class_indicies = [i for i, l in enumerate(self.mapped_labels) if l==label]
        return class_indicies
    
    def class_weights(self):
        """ Returns a list containing the weights of each class.

        Returns:
            list: a list containing the weights of each class.
        """
        class_dist = self.get_data_distribution()
        
        # Given a class number of data points, returns the weight of this class:
        class_weight = lambda x : 1 - (x / len(self.labels))
        
        weights = []
        for value in list(class_dist.values()):
            weight = class_weight(value)
            weights.append(weight)
        
        return weights      