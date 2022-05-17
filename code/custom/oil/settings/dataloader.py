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
        
        img_path = "../data/naverboostcamp_train/JPEGImages/"+ data['file_name']
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
        
        img_path = "../data/naverboostcamp_val/JPEGImages/"+ data['file_name']
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

def get_loader(train_data, valid_data,transform):
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