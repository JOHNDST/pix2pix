import glob
import torch
from PIL import Image
from torch.utils.data import Dataset
import numpy as np


class PM25(Dataset):
    
    def __init__(self,
                 root: str='.',
                 transform=None,
                 download: bool=True,
                 mode: str='train',
                 direction: str='B2A'):
        
        """
        Initialize the Facades dataset.

        Parameters:
        - root (str): Root directory for dataset.
        - transform (callable, optional): A function/transform to apply to the images.
        - download (bool): If true, downloads the dataset.
        - mode (str): Specifies the mode of the dataset ('train' or 'test').
        - direction (str): Direction of the image pairs ('B2A' or 'A2B').
        """        


        self.root=root
        self.files=sorted(glob.glob(f"{root}/pm25/{mode}/*.npy"))
        # print(glob.glob(f"{root}/pm25/{mode}/*.npy"))
        self.transform=transform
        self.download=download
        self.mode=mode
        self.direction=direction
        
    def __len__(self,):
        return len(self.files)
    
    def __getitem__(self, idx):

        """
        Get an image pair from the dataset at the given index.

        Parameters:
        - idx (int): Index of the image to retrieve.

        Returns:
        - (tuple): A tuple containing the two cropped images (imgA, imgB).
        """

        data = np.load(self.files[idx])
        # print(data.shape)
        # Split into imgA (last channel) and imgB (first 9 channels)
        imgA = data[:, :, -1]
        imgB = data[:, :, :9]
        
        # Convert imgA to the shape (H, W, 1) to make it compatible with transformations
        imgA = np.expand_dims(imgA, axis=-1)
        
        if self.transform:
            imgA, imgB = self.transform(imgA, imgB)
            
        if self.direction == 'A2B':
            return imgA, imgB
        else:
            return imgB, imgA