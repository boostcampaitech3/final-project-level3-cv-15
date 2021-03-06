from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import pandas as pd
from PIL import Image
import random

class WrinkletrainDataset(Dataset):
    num_classes = 5
    
    image_paths = []
    part_labels = []
    wrinkle_labels = []
    
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
    def __getitem__(self, idx):
        data=self.df.iloc[idx]
        
        img_path = "/opt/ml/data/naverboostcamp_train/JPEGImages/"+ data['file_name']
        image = Image.open(img_path)
        
        if self.transform:
            image = self.transform(image)
            
        wrinkle_labels = data['wrinkle']
        part_labels = data['part']
            
        return image, wrinkle_labels
    
    def __len__(self):
        return len(self.df)

class WrinklevalDataset(Dataset):
    num_classes = 5
    
    image_paths = []
    part_labels = []
    wrinkle_labels = []
    
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
    def __getitem__(self, idx):
        data=self.df.iloc[idx]
        
        img_path = "/opt/ml/data/naverboostcamp_val/JPEGImages/"+ data['file_name']
        image = Image.open(img_path)
        
        if self.transform:
            image = self.transform(image)
            
        wrinkle_labels = data['wrinkle']
        part_labels = data['part']
            
        return image, wrinkle_labels
    
    def __len__(self):
        return len(self.df)

def getDataloader(train_transform, val_transform, batch, train_worker, valid_worker, weight):
    train_data = pd.read_csv('/opt/ml/data/naverboostcamp_train.csv')
    valid_data = pd.read_csv('/opt/ml/data/naverboostcamp_val.csv')

    train_data = train_data[['part', 'wrinkle', 'file_name']]
    valid_data = valid_data[['part', 'wrinkle', 'file_name']]

    train_data = train_data[train_data['wrinkle'] >= 0]
    valid_data = valid_data[valid_data['wrinkle'] >= 0]

    train_data.reset_index(drop=True, inplace=True)
    valid_data.reset_index(drop=True, inplace=True)

    for score, w in enumerate(weight):
        score_index = train_data[train_data.wrinkle == score].index.tolist()
        if w < 1 : 
            drop_index = random.sample(score_index, int(len(score_index)*(1-w)))
            train_data = train_data.drop(drop_index, axis=0)
            train_data.reset_index(inplace=True, drop=True)

        elif w > 1 :
            Int, Dec = int(w), w - int(w)
            add_index = random.sample(score_index, int(len(score_index)*Dec))
            train_data = pd.concat([train_data, train_data.iloc[score_index * (Int-1) + add_index]])
            train_data.reset_index(inplace=True, drop=True)

    print(train_data.groupby(['wrinkle']).count()['part'])
    
    train_dataset= WrinkletrainDataset(train_data, train_transform)
    val_dataset= WrinklevalDataset(valid_data, val_transform)
    
    train_loader = DataLoader(train_dataset,
                              shuffle=True,
                              num_workers = train_worker,
                              batch_size = batch,)
    valid_loader = DataLoader(val_dataset,
                              shuffle=False,
                              num_workers = valid_worker,
                              batch_size = batch,)
    return train_loader, valid_loader