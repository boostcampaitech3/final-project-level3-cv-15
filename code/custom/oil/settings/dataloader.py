from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
from PIL import Image

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
        image = Image.open(img_path)
        
        if self.transform:
            image = self.transform(image)
            
        oil_labels = data['oil']
        part_labels = data['part']
            
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
        image = Image.open(img_path)
        
        if self.transform:
            image = self.transform(image)
            
        oil_labels = data['oil']
        part_labels = data['part']
            
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