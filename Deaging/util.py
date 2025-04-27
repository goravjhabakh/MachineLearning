import os
import matplotlib.pyplot as plt
from collections import defaultdict

root_dir = '../Datasets/UTKFace/UTKFace'

def get_dict(root):
    ages = defaultdict(int)

    for img_name in os.listdir(root):
        if img_name.endswith('.jpg'):
            age = int(img_name.split('_')[0])
            ages[age] += 1
    print(ages)

def histogram(root):
    ages = []

    for img_name in os.listdir(root):
        if img_name.endswith('.jpg'):
            age = int(img_name.split('_')[0])
            ages.append(age)

    plt.hist(ages, bins=range(0, 100, 5), edgecolor='black')
    plt.xlabel('Age')
    plt.ylabel('Count')
    plt.title('UTKFace Age Distribution')
    plt.show()

def age_count(root):
    import os
import matplotlib.pyplot as plt
from collections import defaultdict
from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

root_dir = '../Datasets/UTKFace/UTKFace'

def get_dict(root):
    ages = defaultdict(int)

    for img_name in os.listdir(root):
        if img_name.endswith('.jpg'):
            age = int(img_name.split('_')[0])
            ages[age] += 1
    print(ages)

def histogram(root):
    ages = []

    for img_name in os.listdir(root):
        if img_name.endswith('.jpg'):
            age = int(img_name.split('_')[0])
            ages.append(age)

    plt.hist(ages, bins=range(0, 100, 5), edgecolor='black')
    plt.xlabel('Age')
    plt.ylabel('Count')
    plt.title('UTKFace Age Distribution')
    plt.show()

def age_count(root):
    young_count = 0
    old_count = 0

    for img_name in os.listdir(root_dir):
        if img_name.endswith('.jpg'):
            age = int(img_name.split('_')[0])
            if 20 < age < 30:
                young_count += 1
            elif age >= 45:
                old_count += 1

    print(f'Young (20-30): {young_count} images')
    print(f'Old (45+): {old_count} images')

age_count(root_dir)