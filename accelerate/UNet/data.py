# Carvana

import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms

class CarvanaDataset(Dataset):
    def __init__(self, 
                 images_dir, 
                 masks_dir, 
                 transform=None, 
                 mask_suffix='_mask'):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.mask_suffix = mask_suffix
        
        # imgs file
        self.ids = [p.stem for p in self.images_dir.glob("*.jpg")] 
        
        self.transform = transform or transforms.Compose([
            transforms.Resize((256, 256)),   
            transforms.ToTensor()           
        ])
        
    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        name = self.ids[idx]
        
        img_path = self.images_dir / f"{name}.jpg"
        image = Image.open(img_path).convert("RGB")  # 3 channels
        image = self.transform(image)  

        mask_path = self.masks_dir / f"{name}{self.mask_suffix}.gif"
        mask = Image.open(mask_path)  
        mask = transforms.Resize((256, 256), interpolation=transforms.InterpolationMode.NEAREST)(mask)
        mask_np = np.array(mask, dtype=np.int64)  
        mask = torch.from_numpy(mask_np).long()   
        
        return {
            'image': image,
            'mask': mask
        }