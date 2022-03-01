import torch
from torchvision import transforms
import pandas as pd
import os
from PIL import Image

class TestDataset(torch.utils.data.Dataset):
    """
    Test dataset.
    """

    def __init__(self):
        # Read csv into labels
        self.labels = pd.read_csv('./data/test.csv')

        # Image directory path
        self.img_dir = './cassava-leaf-classification/train_images'

    def __getitem__(self, index):
        # Read the image into a tensor
        image = Image.open(os.path.join(self.img_dir, self.labels.iloc[index]['image_id']))
        
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
        label = self.labels.iloc[index]['label']
        
        return preprocess(image), label

    def __len__(self):
        return len(self.labels)
