import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from torch.optim.lr_scheduler import StepLR
import random

import os
import gc
import copy

import numpy as np
import pandas as pd

import easydict
from tqdm import notebook

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam, AdamW
from efficientnet_pytorch import EfficientNet

from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

from torch.optim.lr_scheduler import ReduceLROnPlateau

from torchvision import transforms
import timm

######################################################################
class OiltrainDataset(Dataset):
    num_classes = 5
    
    image_paths = []
    part_labels = []
    oil_labels = []
    
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
    def __getitem__(self, idx):
        data=self.df.iloc[idx]
        
        img_path = "../../../data/naverboostcamp_train/JPEGImages/"+ data['file_name']
        image = Image.open(img_path)
        
        if self.transform:
            image = self.transform(image)
            
        oil_labels = data['oil']
        # print(oil_labels)
        part_labels = data['part']
        # multi_class_label = self.encode_multi_class(mask_label, gender_label, age_label)
            
        return image, oil_labels
    
    def __len__(self):
        return len(self.df)

class OilvalDataset(Dataset):
    num_classes = 5
    
    image_paths = []
    part_labels = []
    oil_labels = []
    
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
    def __getitem__(self, idx):
        data=self.df.iloc[idx]
        
        img_path = "../../../data/naverboostcamp_val/JPEGImages/"+ data['file_name']
        image = Image.open(img_path)
        
        if self.transform:
            image = self.transform(image)
            
        oil_labels = data['oil']
        # print(oil_labels)
        part_labels = data['part']
        # multi_class_label = self.encode_multi_class(mask_label, gender_label, age_label)
            
        return image, oil_labels
    
    def __len__(self):
        return len(self.df)
################################################################

def get_loader(train_data, valid_data,transform, args):
    train_dataset= OiltrainDataset(train_data, transform)
    val_dataset= OilvalDataset(valid_data, transform)
    
    train_loader = DataLoader(train_dataset,
                              shuffle=True,
                              num_workers = args.num_workers,
                              batch_size = args.batch_size,)
    valid_loader = DataLoader(val_dataset,
                              shuffle=False,
                              num_workers = args.num_workers,
                              batch_size = args.batch_size,)
    return train_loader, valid_loader

def get_model(args):
    if args.model == 'timm':
        model=Timm(args)
    
    model.to(args.device)
    
    return model

def get_optimizer(model, args):
    if args.optimizer == 'adam':
        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=0.0)
    
    optimizer.zero_grad()
    
    return optimizer

def get_scheduler(optimizer, args):
    if args.scheduler == 'steplr':
        scheduler = StepLR(optimizer, 20, gamma=0.5)
        
    return scheduler

class Timm(nn.Module):
    def __init__(self, args, class_n=5):
        super().__init__()
        self.model = timm.create_model('efficientnet_b4', pretrained=True, num_classes=class_n)
        # self.dropout = nn.Dropout(0.6)
    def forward(self, x):
        x = self.model(x)
        return x
    
def train(args, model, train_loader, optimizer):
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    
    corrects=0
    
    for step,(images,labels) in enumerate(train_loader):
        images = images.to(args.device)
        labels = labels.to(args.device)
        
        outputs= model(images)
        # outputs = outputs[0]
        
        loss = criterion(outputs, labels)
        loss.backward()
        
        optimizer.step()
        optimizer.zero_grad()
        
        if step % args.log_steps == 0:
            print(f"Training steps: {step} Loss : {str(loss.item())}")
            
        _, preds = torch.max(outputs,1)
        
        corrects += torch.sum(preds == labels.data)
    
    acc = corrects / args.train_len
    
    return acc

def valid(args, model, valid_loader, optimizer):
    model.eval()
    
    corrects=0
    
    for images,labels in valid_loader:
        images = images.to(args.device)
        labels = labels.to(args.device)
        
        outputs= model(images)

        _, preds = torch.max(outputs,1)
        corrects += torch.sum(preds == labels.data)
    
    acc = corrects / args.valid_len
    
    print(f'VALID ACC : {acc}\n')
    
    return acc, outputs

def run(args, train_data, valid_data):
    args.train_len = len(train_data)
    args.valid_len = len(valid_data)
    transform = transforms.Compose([transforms.ToTensor(),
                               transforms.Resize((256,256))])
    train_loader, valid_loader = get_loader(train_data, valid_data, transform, args)
    
    model = get_model(args)
    optimizer = get_optimizer(model, args)
    scheduler = get_scheduler(optimizer, args)
    
    best_acc = -1
    
    iter_n = 1
    
    for epoch in notebook.tqdm(range(args.n_epochs)):
        print(f'Epoch {epoch+1}/{args.n_epochs}')
        
        train_acc = train(args, model, train_loader, optimizer)
        
        valid_acc , outputs = valid(args, model, valid_loader, optimizer)
        
        if valid_acc > best_acc :
            best_acc = valid_acc
            
            save_name = f"{args.timm_model}_{str(best_acc.item())[:4]}"
            
            torch.save(model, os.path.join(args.model_path, save_name))
            print(f'model saved! {save_name}')
        
        scheduler.step()

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    
def main():
    
    config = {}

    config['seed'] = 42
    config['device'] = "cuda" if torch.cuda.is_available() else "cpu"
    config['model_path'] =  './model'

    config['num_workers'] = 4
    config['pin_memory']=False

    # config['hidden_dim'] = 128
    config['dropout'] = 0.1

    config['n_epochs']=20
    config['batch_size'] = 16
    config['lr']=1e-3
    config['clip_grad']=10
    config['log_steps']=20
    config['patience']=10

    config['model'] = 'timm'
    config['timm_model'] = 'efficientnet_b4'

    config['optimizer'] = 'adam'
    config['scheduler'] = 'steplr'

    config['load']=False
    config['model_name']=''
    
    args=easydict.EasyDict(config)
    
    train_df = pd.read_csv('../../../data/naverboostcamp_train.csv')
    val_df = pd.read_csv('../../../data/naverboostcamp_val.csv')

    train_df = train_df[['part', 'oil', 'file_name']]
    val_df = val_df[['part', 'oil', 'file_name']]

    train_rm_idx1 = train_df[train_df['oil'] == -1].index
    train_rm_idx2 = train_df[train_df['oil'] == -2].index

    val_rm_idx1 = val_df[val_df['oil'] == -1].index
    val_rm_idx2 = val_df[val_df['oil'] == -2].index

    train_df.drop(train_rm_idx1, inplace=True)
    train_df.drop(train_rm_idx2, inplace=True)

    val_df.drop(val_rm_idx1, inplace=True)
    val_df.drop(val_rm_idx2, inplace=True)

    train_df.reset_index(drop=True, inplace=True)
    val_df.reset_index(drop=True, inplace=True)
    
    transform = transforms.Compose([transforms.ToTensor(),
                               transforms.Resize((256,256))])
    
    # criterion = nn.CrossEntropyLoss()
    
    seed_everything(42)
    run(args, train_df, val_df)
    
if __name__=="__main__":
    main()