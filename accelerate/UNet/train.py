import os
import kagglehub
import matplotlib.pyplot as plt
import numpy as np
import zipfile
import torch
import torch.optim as optim

from pathlib import Path
from PIL import Image
from torchvision import transforms

from UNet import UNet, UNetTrainer
from data import CarvanaDataset


# Carvana

BATCH_SIZE = 1  
NUM_WORKERS = 4  # threads 2~8


data_root = "../../Data/Carvana_segmentation" 
model_save_root = "../../Models/UNet"
os.makedirs(model_save_root,exist_ok=True)


os.environ['KAGGLE_API_TOKEN'] = 'KGAT_965c4d5c15efac968b4eaa90d18f571c'
path = kagglehub.competition_download(
    'carvana-image-masking-challenge', 
    output_dir=data_root
)
print(f"Dataset downloaded to: {path}")


for file in os.listdir(data_root):
    if file.endswith('.zip'):                     
        zip_path = os.path.join(data_root, file)
        extract_dir = os.path.join(data_root, os.path.splitext(file)[0])
        if os.path.exists(extract_dir):
            continue  
        os.makedirs(extract_dir, exist_ok=True)
        print(f"unziping {file}")   
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)     


train_hq_path=os.path.join(data_root,"train_hq","train_hq")
train_mask_zip_path=os.path.join(data_root,"train_masks","train_masks")
test_hq_path=os.path.join(data_root,"test_hq","test_hq")

dir_img = Path(train_hq_path)
dir_mask = Path(train_mask_zip_path)

train_set = CarvanaDataset(dir_img, dir_mask)
train_loader = torch.utils.data.DataLoader(
    dataset=train_set,        
    batch_size=BATCH_SIZE,    
    shuffle=True,             
    num_workers=NUM_WORKERS,  
    pin_memory=True,          # speeds up CPU-GPU batch transfer
    drop_last=False           
)


net=UNet(num_class=2)

lr=1e-5
optimizer = optim.Adam(net.parameters(), lr=lr)

trainer=UNetTrainer(model=net,
                    optimizer=optimizer)

trainer.train(num_epochs=5,
              train_dataloader=train_loader,
              save_dir=os.path.join(model_save_root,"UNet.pth"),
              log_interval=50)
