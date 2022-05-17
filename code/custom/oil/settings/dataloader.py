from torch.utils.data.dataloader import DataLoader

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
        
        img_path = "../../../../data/naverboostcamp_val/JPEGImages/"+ data['file_name']
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

def getDataoader(train_transform, val_transform, batch, train_workder, valid_worker):
    train_df = pd.read_csv('../../../../data/naverboostcamp_train.csv')
    val_df = pd.read_csv('../../../../data/naverboostcamp_val.csv')

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
    
    train_dataset= OiltrainDataset(train_data, transform)
    val_dataset= OilvalDataset(valid_data, transform)
    
    train_loader = DataLoader(train_dataset,
                              shuffle=True,
                              num_workers = train_workers,
                              batch_size = batch,)
    valid_loader = DataLoader(val_dataset,
                              shuffle=False,
                              num_workers = valid_workers,
                              batch_size = batch,)
    return train_loader, valid_loader