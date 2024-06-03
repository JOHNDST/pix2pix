import glob
import torch
from PIL import Image
from torch.utils.data import Dataset
from .utils import download_and_extract

class Facades(Dataset):
    url="https://github.com/akanametov/pix2pix/releases/download/1.0/facades.zip"
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

        if download:
            _ = download_and_extract(root, self.url)
        self.root=root
        self.files=sorted(glob.glob(f"{root}/facades/{mode}/*.jpg"))
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

        img = Image.open(self.files[idx]).convert('RGB')
        W, H = img.size
        cW = W//2
        imgA = img.crop((0, 0, cW, H))
        imgB = img.crop((cW, 0, W, H))
        
        if self.transform:
            imgA, imgB = self.transform(imgA, imgB)
            
        if self.direction == 'A2B':
            return imgA, imgB
        else:
            return imgB, imgA