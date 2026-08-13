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


def plot_img_and_mask(img, mask):
    if isinstance(img, torch.Tensor):
        img = img.squeeze(0).cpu().numpy()   # 若 batch 维度存在
        img = np.transpose(img, (1, 2, 0))   # 从 C,H,W → H,W,C
    classes = mask.max() + 1
    fig, ax = plt.subplots(1, classes + 1)
    ax[0].set_title('Input image')
    ax[0].imshow(img)
    for i in range(classes):
        ax[i + 1].set_title(f'Mask (class {i + 1})')
        ax[i + 1].imshow(mask == i)
    plt.xticks([]), plt.yticks([])
    plt.show()



net=UNet(num_class=2).to(DEVICE)
state_dict=torch.load(os.path.join(model_save_root,"UNet.pth"), map_location=DEVICE)
mask_values = state_dict.pop('mask_values', [0, 1])
net.load_state_dict(state_dict)
net.eval()

transform = transforms.Compose([
            transforms.Resize((256, 256)),   
            transforms.ToTensor()           
            ])

test_imgs_list=os.listdir(test_hq_path)[0:1]

for i, filename in enumerate(test_imgs_list):
    img_path = os.path.join(test_hq_path, filename)
    img = Image.open(img_path).convert("RGB")
    img=transform(img)
    img = img.unsqueeze(0)
    img = img.to(device=DEVICE, dtype=torch.float32)

    mask=None

    with torch.no_grad():
        output = net(img).cpu()
        if net.num_class > 1:
            mask = output.argmax(dim=1)
        else:
            mask = torch.sigmoid(output) > 0.5

    mask=mask[0].long().squeeze().numpy()

    plot_img_and_mask(img, mask)