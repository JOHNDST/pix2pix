import PIL
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import os

class Transformer(object):
    """"Base class for transforms"""
    def __init__(self,):
        pass
    def __call__(self, imgA, imgB=None):
        pass

class Compose(Transformer):
    """Compose multiple transforms"""
    def __init__(self, transforms=[]):
        super().__init__()
        self.transforms = transforms
        
    def __call__(self, imgA, imgB=None):
        if imgB is None:
            for transform in self.transforms:
                imgA = transform(imgA, imgB)
            return imgA
        for transform in self.transforms:
            imgA, imgB = transform(imgA, imgB)
        return imgA, imgB
    
class Resize(Transformer):
    """Resize imageA and imageB"""
    def __init__(self, size=(256, 256)):
        super().__init__()
        self.size = size
        
    def __call__(self, imgA, imgB=None):
        imgA = np.array([np.array(Image.fromarray(imgA[:,:,i]).resize(self.size)) for i in range(imgA.shape[2])]).transpose(1, 2, 0)
        if imgB is None:
            return imgA
        imgB = np.array([np.array(Image.fromarray(imgB[:,:,i]).resize(self.size)) for i in range(imgB.shape[2])]).transpose(1, 2, 0)
        return imgA, imgB
    
class CenterCrop(Transformer):
    """Center crop imageA and imageB"""
    def __init__(self, size=(256, 256), p=0.5):
        super().__init__()
        self.size = size
        self.p = p
        
    def __call__(self, imgA, imgB=None):
        H, W = imgA.shape[:2]
        cH, cW = self.size
        top = (H - cH) // 2
        left = (W - cW) // 2
        if np.random.uniform() < self.p:
            imgA = imgA[top:top + cH, left:left + cW]
            if imgB is not None:
                imgB = imgB[top:top + cH, left:left + cW]
        else:
            imgA = np.array([np.array(Image.fromarray(imgA[:,:,i]).resize((cW, cH))) for i in range(imgA.shape[2])]).transpose(1, 2, 0)
            if imgB is not None:
                imgB = np.array([np.array(Image.fromarray(imgB[:,:,i]).resize((cW, cH))) for i in range(imgB.shape[2])]).transpose(1, 2, 0)
        if imgB is None:
            return imgA
        return imgA, imgB
    
class Rotate(Transformer):
    """Rotate imageA and imageB"""
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        
    def __call__(self, imgA, imgB=None):
        if np.random.uniform() < self.p:
            imgA = np.rot90(imgA, 2)  # Rotate by 180 degrees
            if imgB is not None:
                imgB = np.rot90(imgB, 2)
        if imgB is None:
            return imgA
        return imgA, imgB
    
class HorizontalFlip(Transformer):
    """Horizontally flip imageA and imageB"""
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        
    def __call__(self, imgA, imgB=None):
        if np.random.uniform() < self.p:
            imgA = np.fliplr(imgA).copy()
            if imgB is not None:
                imgB = np.fliplr(imgB).copy()
        if imgB is None:
            return imgA
        return imgA, imgB
    
class VerticalFlip(Transformer):
    """Vertically flip imageA and imageB"""
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p
        
    def __call__(self, imgA, imgB=None):
        if np.random.uniform() < self.p:
            imgA = np.flipud(imgA).copy()
            if imgB is not None:
                imgB = np.flipud(imgB).copy()
        if imgB is None:
            return imgA
        return imgA, imgB
    
class ToTensor(Transformer):
    """Convert imageA and imageB to torch.tensor"""
    def __init__(self):
        super().__init__()
    
    def __call__(self, imgA, imgB=None):
        imgA = torch.from_numpy(imgA).float().permute(2, 0, 1) / 255.0
        if imgB is None:
            return imgA
        imgB = torch.from_numpy(imgB).float().permute(2, 0, 1) / 255.0
        return imgA, imgB
    
class Normalize(Transformer):
    """Normalize imageA and imageB"""
    def __init__(self, meanA, stdA, meanB, stdB):
        """
        :param meanA: list, means for each channel in imgA
        :param stdA: list, stds for each channel in imgA
        :param meanB: list, means for each channel in imgB
        :param stdB: list, stds for each channel in imgB
        """
        super().__init__()
        self.meanA = torch.tensor(meanA).view(-1, 1, 1)
        self.stdA = torch.tensor(stdA).view(-1, 1, 1)
        self.meanB = torch.tensor(meanB).view(-1, 1, 1)
        self.stdB = torch.tensor(stdB).view(-1, 1, 1)
        
    def __call__(self, imgA, imgB=None):
        imgA = (imgA - self.meanA) / self.stdA
        if imgB is not None:
            imgB = (imgB - self.meanB) / self.stdB
        if imgB is None:
            return imgA
        return imgA, imgB

class DeNormalize(Transformer):
    """DeNormalize imageA and imageB"""
    def __init__(self, mean, std):
        super().__init__()
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)
        
    def __call__(self, imgA, imgB=None):
        imgA = imgA * self.std + self.mean
        imgA = torch.clip(imgA, 0., 1.)
        if imgB is not None:
            imgB = imgB * self.std + self.mean
            imgB = torch.clip(imgB, 0., 1.)
        if imgB is None:
            return imgA
        return imgA, imgB

# Define your custom transformation
class CustomTransform(Transformer):
    def __call__(self, imgA, imgB=None):
        # Example transformation logic (this should be customized as needed)
        imgA = imgA / 255.0  # Normalize to [0, 1]
        if imgB is not None:
            imgB = imgB / 255.0  # Normalize to [0, 1]
        if imgB is None:
            return imgA
        return imgA, imgB
# Creating the dataset
class CustomDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.files = sorted([os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.npy')])
        
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        """
        Get an image pair from the dataset at the given index.

        Parameters:
        - idx (int): Index of the image to retrieve.

        Returns:
        - (tuple): A tuple containing the two images (imgA, imgB).
        """
        # Load the 10-channel numpy array
        data = np.load(self.files[idx])
        
        # Split into imgA (last channel) and imgB (first 9 channels)
        imgA = data[:, :, -1]
        imgB = data[:, :, :9]
        
        # Convert imgA to the shape (H, W, 1) to make it compatible with transformations
        imgA = np.expand_dims(imgA, axis=-1)
        
        if self.transform:
            imgA, imgB = self.transform(imgA, imgB)
        
        return imgA, imgB
