import os
from collections import defaultdict
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

root_dir = '../Datasets/UTKFace/UTKFace'

class DeagingDataset(Dataset):
    def __init__(self, root, old = (50,75), young = (15,25), transform = None):
        super().__init__()
        self.root = root

    def get_unique_ages(self):
        ages = defaultdict(int)
        for file in os.listdir(self.root):
            age = int(file.split('_')[0])
            ages[age]+=1
        return ages
    
dataset = DeagingDataset(root_dir)
ages = dataset.get_unique_ages()
ages = sorted(ages.items(), key=lambda x: x[1], reverse=True)
print(ages)