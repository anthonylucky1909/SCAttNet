import os
import cv2
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import Dataset

class SuperResolutionImageSet(Dataset):
    def __init__(self, data_folder, lr_augment, hr_augment, joint_augment):
        super().__init__()
        self.data_path = data_folder
        self.lr_preprocess = lr_augment
        self.hr_preprocess = hr_augment
        self.joint_preprocess = joint_augment
        self.file_names = [f for f in os.listdir(data_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.data_path, self.file_names[idx])
        img_data = cv2.imread(img_path)
        img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)
        
        augmented = self.joint_preprocess(image=img_data)
        img_data = augmented['image']
        
        lr_img = self.lr_preprocess(image=img_data)['image']
        hr_img = self.hr_preprocess(image=img_data)['image']
        
        return lr_img, hr_img

def get_transforms(high_res, low_res):
    hr_preprocessing = A.Compose([
        A.Resize(width=high_res, height=high_res),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
        ToTensorV2(),
    ])

    lr_preprocessing = A.Compose([
        A.Resize(width=low_res, height=low_res, interpolation=Image.BICUBIC),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
        ToTensorV2(),
    ])

    augmentation_pipeline = A.Compose([
        A.RandomCrop(width=high_res, height=high_res),
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
    ])
    
    return hr_preprocessing, lr_preprocessing, augmentation_pipeline

def get_dataloaders(config):
    hr_preprocess, lr_preprocess, joint_augment = get_transforms(config.high_res, config.low_res)
    
    train_dataset = SuperResolutionImageSet(
        data_folder=config.data_paths['train'],
        lr_augment=lr_preprocess,
        hr_augment=hr_preprocess,
        joint_augment=joint_augment
    )
    
    test_datasets = {
        name: SuperResolutionImageSet(
            data_folder=path,
            lr_augment=lr_preprocess,
            hr_augment=hr_preprocess,
            joint_augment=joint_augment
        )
        for name, path in config.data_paths.items() if name != 'train'
    }
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        num_workers=config.num_workers,
        batch_size=config.batch_size,
        shuffle=True,
        pin_memory=True
    )
    
    return train_loader, test_datasets