import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

transform = transforms.Compose([
    transforms.Resize((112, 112)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

class ImageClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layer_1 = nn.Sequential(
          nn.Conv2d(3, 64, 3),
          nn.ReLU(),
          nn.BatchNorm2d(64),
          nn.MaxPool2d(2))

        self.conv_layer_2 = nn.Sequential(
          nn.Conv2d(64, 128, 3),
          nn.ReLU(),
          nn.BatchNorm2d(128),
          nn.MaxPool2d(2))

        self.conv_layer_3 = nn.Sequential(
          nn.Conv2d(128, 128, 3),
          nn.ReLU())

        self.classifier = nn.Sequential(
          nn.Flatten(),
          nn.Linear(in_features=128*24*24, out_features=2))

    def forward(self, x: torch.Tensor):
        x = self.conv_layer_1(x)
        x = self.conv_layer_2(x)
        x = self.conv_layer_3(x)
        x = self.classifier(x)
        return x


