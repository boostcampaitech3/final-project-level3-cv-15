from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
from PIL import Image
import cv2
import albumentations as A
import numpy as np

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
        img_path = "/opt/ml/data/naverboostcamp_train/JPEGImages/"+ data['file_name']

        oil_labels = data['oil']
        part_labels = data['part']
        # image = Image.open(img_path)
        image = cv2.imread(img_path)
        image = A.Resize(always_apply=False, p=1.0, height=500, width=700, interpolation=0)(image = image)
        image = image["image"]
        if int(part_labels) == 1:
            image = cv2.rectangle(image, (0,0), (700,120), (0,0,0),-1)
            image = cv2.rectangle(image, (0,400), (700,500), (0,0,0),-1)
            polly1 = np.array([[200,100],[0,100],[0,400]])
            polly2 = np.array([[500,100],[700,100],[700,400]])
            image = cv2.fillPoly(image,[polly1],color=(0,0,0))
            image = cv2.fillPoly(image,[polly2],color=(0,0,0))

        if self.transform:
            image = self.transform(image)
            
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
        
        img_path = "/opt/ml/data/naverboostcamp_val/JPEGImages/"+ data['file_name']
        # image = Image.open(img_path)
        oil_labels = data['oil']
        part_labels = data['part']

        image = cv2.imread(img_path)
        image = A.Resize(always_apply=False, p=1.0, height=500, width=700, interpolation=0)(image = image)
        image = image["image"]
        # if int(part_labels) == 1:
        #     image = cv2.rectangle(image, (0,0), (700,120), (0,0,0),-1)
        #     image = cv2.rectangle(image, (0,400), (700,500), (0,0,0),-1)
        #     polly1 = np.array([[200,100],[0,100],[0,400]])
        #     polly2 = np.array([[500,100],[700,100],[700,400]])
        #     image = cv2.fillPoly(image,[polly1],color=(0,0,0))
        #     image = cv2.fillPoly(image,[polly2],color=(0,0,0))

        
        if self.transform:
            image = self.transform(image)
            
        
            
        return image, oil_labels
    
    def __len__(self):
        return len(self.df)

def getDataloader(train_transform, val_transform, batch, train_worker, valid_worker):
    train_data = pd.read_csv('/opt/ml/data/naverboostcamp_train.csv')
    valid_data = pd.read_csv('/opt/ml/data/naverboostcamp_val.csv')

    train_data = train_data[['part', 'oil', 'file_name']]
    valid_data = valid_data[['part', 'oil', 'file_name']]

    train_data = train_data[train_data['oil'] >= 0]
    valid_data = valid_data[valid_data['oil'] >=0]
 

    train_data.reset_index(drop=True, inplace=True)
    valid_data.reset_index(drop=True, inplace=True)
    
    train_dataset= OiltrainDataset(train_data, train_transform)
    val_dataset= OilvalDataset(valid_data, val_transform)
    
    train_loader = DataLoader(train_dataset,
                              shuffle=True,
                              num_workers = train_worker,
                              batch_size = batch,)
    valid_loader = DataLoader(val_dataset,
                              shuffle=False,
                              num_workers = valid_worker,
                              batch_size = batch,)
    return train_loader, valid_loader