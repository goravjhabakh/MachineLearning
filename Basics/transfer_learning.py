# Imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tqdm import tqdm

# Hyperparameters
input_size = 32
in_channels = 3
num_classes = 10
lr = 0.001
batch_size = 64
num_epochs = 5

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load pretrained model
model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
model.fc = nn.Sequential(
    nn.Linear(512, 100),
    nn.ReLU(),
    nn.Linear(100,num_classes)
)
model.to(device)

# Load Data
train_dataset = datasets.CIFAR10(root='../Datasets/', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = datasets.CIFAR10(root='../Datasets/', train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# Train Netowork
for epoch in range(num_epochs):
    correct = 0
    total = 0
    loop = tqdm(train_loader, desc=f'Epoch: [{epoch+1}/{num_epochs}]')

    for data,target in loop:
        # Format data into correct shape
        data, target = data.to(device), target.to(device)

        # Forward
        output = model(data)
        loss = criterion(output,target)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accuracy
        _, pred = output.max(1)  # val,ind
        correct += (pred == target).sum()
        total += pred.size(0)
        acc = correct / total * 100

    print(f'Loss: {loss.item():.4f}, Accuracy: {acc:.2f}%')

# Check accuracy on test dataset
correct = 0
total = 0
model.eval()

with torch.no_grad():
    loop = tqdm(test_loader, desc='Evaluating... ')
    for x,y in loop:
        x, y = x.to(device), y.to(device)

        output = model(x)
        _, pred = output.max(1)  # val,ind
        correct += (pred == y).sum()
        total += pred.size(0)
        acc = correct / total * 100
    print(f'Accuracy: {acc:.2f}%')

# No pretrained weights
# Train loss = 0.7844, accuracy = 76.48%
# Test accuracy = 66.75%
#
# Freezing Weights except last 
# Train loss = 1.4307, accuracy = 49.52%
# Test accuracy = 49.40%
#
# With Pretrained weights
# Train loss = 0.6656, accuracy = 88.90%
# Test accuracy = 77.72%