import torch
import torchvision
import pandas as pd
import os

class TestDataset(torch.utils.data.Dataset):
    """
    Dataset that contains 100000 3x224x224 black images (all zeros).
    """

    def __init__(self, transform = None, target_transform = None):
        # Read csv into labels
        self.labels = pd.read_csv('./cassava-leaf-disease-classification/test.csv')

        # Image directory path
        self.img_dir = './cassava-leaf-disease-classification/train_images'

        self.transform, self.target_transform = transform, target_transform

    def __getitem__(self, index):
        # Read the image into a tensor
        image = torchvision.io.read_image(os.path.join(self.img_dir, self.labels.iloc[index]['image_id'])).float()
        
    
        # Do the transforms if they exist
        if self.transform is not None:
            image = self.transform(image)
        label = self.labels.iloc[index]['label']
        if self.target_transform is not None:
            label = self.transform(label)
        
        return image, label

    def __len__(self):
        return len(self.labels)
