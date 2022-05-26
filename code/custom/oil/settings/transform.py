import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision import transforms

# https://albumentations-demo.herokuapp.com/

def getTransform(_p, _limit):
    train_transform = A.Compose([
        A.CLAHE(always_apply=False, p=1.0 , clip_limit=(4, 4), tile_grid_size=(8, 8)),
        A.GridDistortion(always_apply=False, p=0.3, num_steps=5, distort_limit=(-0.3, 0.3), interpolation=0, border_mode=0, value=(0, 0, 0), mask_value=None),
        A.HorizontalFlip(always_apply=False, p=0.5),
        A.CoarseDropout(always_apply=False, p=0.3, max_holes=14, max_height=8, max_width=8, min_holes=14, min_height=8, min_width=8),
        A.GaussNoise(always_apply=False, p=1.0, var_limit=(46.0, 115.0)),
                            ToTensorV2()
                            ])
    
    val_transform = A.Compose([
        A.CLAHE(always_apply=False, p=1.0 , clip_limit=(4, 4), tile_grid_size=(8, 8)),
                            ToTensorV2()
                            ])

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