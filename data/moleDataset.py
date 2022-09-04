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
    def __init__(self, dataset_path=None, labels_path=None, transform=None):
        self.dataset_path = dataset_path
        self.labels_path = labels_path
        self.transform = transform
        
        self.images_paths = self.read_data()
        self.metadata = self.load_metadata()
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
        join_with_data_path = lambda x : os.path.join(self.dataset_path, x)
        
        images_paths = os.listdir(self.dataset_path)
        images_paths = list(map(join_with_data_path, images_paths))
        
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
        # label = metadata['dx'][0]
        
        return image, self.mapped_labels[index], metadata
    
    
    def __len__(self):
        return len(self.images_paths)
    
    
    def get_ground_truth(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        COLUMN_NAME = 'image_id'

        metadata = pd.read_csv(self.labels_path)
        image_ids = [os.path.basename(os.path.normpath(im)).split('.')[0].strip() for im in self.images_paths]
        image_metadata = np.array([metadata.loc[metadata[COLUMN_NAME] == i].to_numpy() for i in image_ids]).squeeze(axis=1)
        labels = image_metadata[:, 2]
        
        unique_values = np.unique(labels)
        # The mapping of the categorical values to numerical values:
        mapping = dict(zip(unique_values, list(range(len(unique_values)))))
        
        # Creating the mapped labels:
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
        
        