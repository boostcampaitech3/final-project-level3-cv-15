from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import timm

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models

class Timm(nn.Module):
    def __init__(self, class_n=5 , model_n='efficientnet_b4'):
        super().__init__()
        self.model = timm.create_model(model_n, pretrained=True, num_classes=class_n)
        in_ch = self.model.classifier.in_features
        self.model.classifier = nn.Identity()

        self.fc1 = nn.Linear(in_ch, in_ch//2)
        self.bn1 = nn.BatchNorm1d(in_ch//2)
        self.fc2 = nn.Linear(in_ch//2, in_ch//4)
        self.bn2 = nn.BatchNorm1d(in_ch//4)
        self.fc3 = nn.Linear(in_ch//4, class_n)

        self.act = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = self.model(x)
        x = self.dropout(self.act(self.bn1(self.fc1(x))))
        x = self.dropout(self.act(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        
        return x

class EfficientNet_b4(nn.Module):
    def __init__(self, class_n=5) -> None:
        super().__init__()
        self.fc = nn.Linear(1792, class_n, bias=True)
        torch.nn.init.kaiming_normal_(self.fc.weight)
        torch.nn.init.zeros_(self.fc.bias)

        self.effnetb4 = models.efficientnet_b4(pretrained=True)
        self.effnetb4.classifier = nn.Sequential(
            nn.Dropout(p=0.4, inplace=True),
            self.fc
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.effnetb4(x)


    
def getModel(modeltype, device, model_n):
    if modeltype == 'timm':
        model=Timm(model_n = model_n)
    elif modeltype == 'efficientb4':
        model=EfficientNet_b4()
    
    model.to(device)
    
    return model

def getRegressionModel(modeltype, device, model_n):
    if modeltype == 'timm':
        model = Timm(class_n = 1, model_n = model_n)
    elif modeltype == 'efficientb4':
        model = EfficientNet_b4(class_n = 1)
    
    model.to(device)
    return model