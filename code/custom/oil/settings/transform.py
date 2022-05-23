import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import transforms

# https://albumentations-demo.herokuapp.com/

def getTransform():
    train_transform = transforms.Compose([transforms.ToTensor(),
                               transforms.Resize((256,256)),])
                            #    transforms.Normalize(mean = [0.6580, 0.5347, 0.4624], std = [0.1837, 0.1616, 0.1509])])
    
    val_transform = transforms.Compose([transforms.ToTensor(),
                               transforms.Resize((256,256)),])
                            #    transforms.Normalize(mean = [0.6580, 0.5347, 0.4624], std = [0.1837, 0.1616, 0.1509])])
    
    return train_transform, val_transform




#   train_transform = \
#     A.Compose([
#       A.Normalize(),
#       A.OneOf([
#         A.Flip(p=1),
#         A.RandomRotate90(p=1)
#       ], p=0.9),
#       # A.OneOf([
#       #   A.RandomBrightnessContrast(0.1,0.1,p=1),
#       #   A.ChannelShuffle(p=1),
#       #   A.HueSaturationValue(p=1)
#       # ], p=0.7),
#       A.RandomResizedCrop(512,512,(0.5,0.9), p=0.8),
#       ToTensorV2(),
#       ])

#   val_transform = A.Compose([
#                             A.Normalize(),
#                             ToTensorV2(),
#                             ])