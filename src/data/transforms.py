# File: src/data/transforms.py
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

def get_train_transforms(img_size: int = 512) -> A.Compose:
    """
    Training augmentation pipeline. 
    Focuses on spatial invariance and slight color jittering to prevent overfitting,
    without distorting the actual pathological geometry.
    """
    return A.Compose([
        A.RandomResizedCrop(
            height=img_size, width=img_size, 
            scale=(0.85, 1.0), ratio=(0.9, 1.1), 
            interpolation=cv2.INTER_CUBIC, p=1.0
        ),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02, p=0.5),
        # Standard ImageNet normalization as expected by timm backbones
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

def get_val_transforms(img_size: int = 512) -> A.Compose:
    """
    Validation/Testing pipeline. Strictly resizing and normalization.
    """
    return A.Compose([
        A.Resize(height=img_size, width=img_size, interpolation=cv2.INTER_CUBIC),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])