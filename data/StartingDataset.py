import torch
import os
import pandas as pd
from pathlib import Path
from PIL import Image
from torchvision import transforms

class StartingDataset(torch.utils.data.Dataset):

    # spits out image tensors which resnet18 should like

    def __init__(self, transform = None, target_transform = None):
        self.root_dir = Path.cwd()

        # Read csv into labels
        self.labels = pd.read_csv(os.path.join(self.root_dir, 'data', 'train.csv'))
        
        # Image directory path
        self.img_dir = os.path.join(self.root_dir, 'cassava-leaf-disease-classification', 'train_images')

        self.transform, self.target_transform = transform, target_transform

    def __getitem__(self, index):
        # Read the image into a tensor dim 224 x 224 x 3
        image = Image.open(os.path.join(self.img_dir, self.labels.iloc[index]['image_id']))
        
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        input_batch = preprocess(image)
    
        label = int(self.labels.iloc[index]['label'])
        
        return input_batch, label

    def __len__(self):
        return len(self.labels)

# print(StartingDataset()[0])
