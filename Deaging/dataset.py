import os
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

root_dir = '../Datasets/UTKFace/UTKFace'

class DeagingDataset(Dataset):
    def __init__(self, root, transform = None):
        super().__init__()
        self.root = root
        self.transform = transform
        self.imgs = []

        for file in os.listdir(root):
            age = int(file.split('_')[0])
            path = os.path.join(root,file)
            if 20 <= age <= 30: self.imgs.append((path,0))
            elif age >= 50: self.imgs.append((path,1))

    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, index):
        img_path, label = self.imgs[index]
        img = Image.open(img_path).convert('RGB')
        if self.transform: img = self.transform(img)
        return img,label