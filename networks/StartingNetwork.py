import torch
import torch.nn as nn

class StartingNetwork(torch.nn.Module):
    """
    Basic logistic regression on 224 x 224 x 3 images.
    """

    def __init__(self):
        super().__init__()
        self.resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained = True)
        self.resnet = torch.nn.Sequential(*(list(self.resnet.children())[:-1])) # output dimension 512
        
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, 5)

    def forward(self, x):
        x = self.resnet(x)
        x = self.fc1(x)
        x = torch.squeeze(x)
        x = self.fc2(x)
        return x
