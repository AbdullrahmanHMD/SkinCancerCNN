from torch.utils.data import Dataset
import os
import pandas as pd
import json
import cv2

class MoleDataset(Dataset):
    """_summary_

    Args:
        Dataset (_type_): _description_
    """
    def __init__(self, dataset_path=None, labels_path=None, test_prop=0.2, test=False):
        self.dataset_path = dataset_path
        self.labels_path = labels_path
        self.test_prop = test_prop
        self.test = test
        self.images_paths = self.read_data()
        self.metadata = self.load_metadata()
    
    
    def read_data(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        
        # If the dataset path is not provided then read from the dataset_location.txt
        # file, otherwise use the value provided when creating the MoleDataset object.
        if self.dataset_path is None:
            data_loc_path = os.path.join(os.getcwd(), "dataset_location.txt")
            
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
            data_loc_path = os.path.join(os.getcwd(), "dataset_location.txt")
            
            with open(data_loc_path, 'r') as file:
                data_dict = json.loads(file.read())
                
            self.labels_path = data_dict['labels']
        
        metadata = pd.read_csv(self.labels_path)
        return metadata

    
    def __getitem__(self, index):
        
        # Getting the image path and reading it via cv2:
        image_path = self.images_paths[index]
        image = cv2.imread(image_path)
        
        # Getting the label and metadata of the given image:
        image_id = os.path.basename(os.path.normpath(image_path)).split('.')[0]
        COLUMN_NAME = 'image_id'
        metadata = self.metadata.loc[self.metadata[COLUMN_NAME] == image_id].to_dict('list')
        label = metadata['dx'][0]
        
        return image, label, metadata
    
    def __len__(self):
        return len(self.images_paths)
    
    def get_ground_truth(self):
        """_summary_

        Returns:
            ndarray: A numpy array containing the labels of all the images.
        """
        COLUMN_NAME = 'image_id'

        metadata = pd.read_csv(self.labels_path)
        
        image_ids = [os.path.basename(os.path.normpath(im)).split('.')[0] for im in self.images_paths]        
        image_metadata = [metadata.loc[metadata[COLUMN_NAME] == id] for id in image_ids]
        labels = image_metadata.loc[:, COLUMN_NAME].to_numpy()
        
        return labels