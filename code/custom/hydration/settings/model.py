from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import timm

import torch
import torch.nn as nn

class Timm(nn.Module):
    def __init__(self, class_n=5):
        super().__init__()
        self.model = timm.create_model('efficientnet_b4', pretrained=True, num_classes=class_n)
        # self.dropout = nn.Dropout(0.6)
        
    def forward(self, x):
        x = self.model(x)
        return x
    
def getModel(modeltype, device):
    if modeltype == 'timm':
        model=Timm()
    
    model.to(device)
    
    return model