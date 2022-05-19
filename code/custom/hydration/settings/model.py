from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
import timm
from resnest.torch import resnest101

import geffnet

import torch
import torch.nn as nn

sigmoid = nn.Sigmoid()

class Swish(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * sigmoid(i)
        ctx.save_for_backward(i)
        return result
    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))

class Swish_Module(nn.Module):
    def forward(self, x):
        return Swish.apply(x)

class multigeff(nn.Module):
    def __init__(self, class_n=5, n_meta_features = 1, n_meta_dim = [16, 8]):
        super().__init__()
        self.n_meta_features = n_meta_features
        self.model = geffnet.create_model('tf_efficientnet_b4_ns', pretrained=True)
        in_ch = self.model.classifier.in_features
        self.dropouts = nn.ModuleList([
            nn.Dropout(0.5) for _ in range(5)
        ])
        if n_meta_features > 0:
            self.meta = nn.Sequential(
                nn.Linear(n_meta_features, n_meta_dim[0]),
                nn.BatchNorm1d(n_meta_dim[0]),
                Swish_Module(),
                nn.Dropout(p=0.3),
                nn.Linear(n_meta_dim[0], n_meta_dim[1]),
                nn.BatchNorm1d(n_meta_dim[1]),
                Swish_Module(),
            )
            in_ch += n_meta_dim[1]
        self.myfc = nn.Linear(in_ch, class_n)
        self.model.classifier = nn.Identity()

        # self.dropout = nn.Dropout(0.6)
    def extract(self, x):
        x = self.model(x)
        return x

    def forward(self, x, x_meta):
        x = self.extract(x).squeeze(-1).squeeze(-1)
        if self.n_meta_features > 0:
            x_meta = self.meta(x_meta)
            x = torch.cat((x, x_meta), dim=1)
        for i, dropout in enumerate(self.dropouts):
            if i == 0:
                out = self.myfc(dropout(x))
            else:
                out += self.myfc(dropout(x))
        out /= len(self.dropouts)
        return out

class multiTimm(nn.Module):
    def __init__(self, class_n=5, n_meta_features = 1, n_meta_dim = [16, 8]):
        super().__init__()
        self.n_meta_features = n_meta_features
        self.model = timm.create_model('efficientnet_b4', pretrained=True, num_classes=class_n)
        in_ch = self.model.classifier.in_features
        self.dropouts = nn.ModuleList([
            nn.Dropout(0.5) for _ in range(5)
        ])
        if n_meta_features > 0:
            self.meta = nn.Sequential(
                nn.Linear(n_meta_features, n_meta_dim[0]),
                nn.BatchNorm1d(n_meta_dim[0]),
                Swish_Module(),
                nn.Dropout(p=0.3),
                nn.Linear(n_meta_dim[0], n_meta_dim[1]),
                nn.BatchNorm1d(n_meta_dim[1]),
                Swish_Module(),
            )
            in_ch += n_meta_dim[1]
        self.myfc = nn.Linear(in_ch, class_n)
        self.model.classifier = nn.Identity()

        # self.dropout = nn.Dropout(0.6)
    def extract(self, x):
        x = self.model(x)
        return x

    def forward(self, x, x_meta):
        x = self.extract(x).squeeze(-1).squeeze(-1)
        if self.n_meta_features > 0:
            x_meta = self.meta(x_meta)
            x = torch.cat((x, x_meta), dim=1)
        for i, dropout in enumerate(self.dropouts):
            if i == 0:
                out = self.myfc(dropout(x))
            else:
                out += self.myfc(dropout(x))
        out /= len(self.dropouts)
        return out

class multiResnest(nn.Module):
    def __init__(self, class_n=5, n_meta_features = 1, n_meta_dim = [16, 8], pretrained = True):
        super().__init__()
        self.n_meta_features = n_meta_features
        self.model = torch.hub.load('zhanghang1989/ResNeSt', 'resnest101', pretrained=True)
        self.dropouts = nn.ModuleList([
            nn.Dropout(0.5) for _ in range(5)
        ])
        in_ch = self.model.fc.in_features
        if n_meta_features > 0:
            self.meta = nn.Sequential(
                nn.Linear(n_meta_features, n_meta_dim[0]),
                nn.BatchNorm1d(n_meta_dim[0]),
                Swish_Module(),
                nn.Dropout(p=0.3),
                nn.Linear(n_meta_dim[0], n_meta_dim[1]),
                nn.BatchNorm1d(n_meta_dim[1]),
                Swish_Module(),
            )
            in_ch += n_meta_dim[1]
        self.myfc = nn.Linear(in_ch, class_n)
        self.model.fc = nn.Identity()

        # self.dropout = nn.Dropout(0.6)
    def extract(self, x):
        x = self.model(x)
        return x

    def forward(self, x, x_meta):
        x = self.extract(x).squeeze(-1).squeeze(-1)
        if self.n_meta_features > 0:
            x_meta = self.meta(x_meta)
            x = torch.cat((x, x_meta), dim=1)
        for i, dropout in enumerate(self.dropouts):
            if i == 0:
                out = self.myfc(dropout(x))
            else:
                out += self.myfc(dropout(x))
        out /= len(self.dropouts)
        return out

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
    elif modeltype == 'multiTimm':
        model = multiTimm()
    elif modeltype == 'multiResnest':
        model = multiResnest()
    elif modeltype == 'giff':
        model = multigeff()
    
    model.to(device)
    
    return model

