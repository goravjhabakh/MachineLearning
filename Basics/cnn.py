# Imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tqdm import tqdm

# Create Convulutional Neural Network
class CNN(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(in_features=16*7*7, out_features=num_classes)

    def forward(self,x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.shape[0],-1)
        x = self.fc1(x)
        return x

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
input_size = 784 # 28*28 for MNIST
num_classes = 10
lr = 0.001
batch_size = 64
num_epochs = 5

# Load Data
train_dataset = datasets.MNIST(root='../Datasets/', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = datasets.MNIST(root='../Datasets/', train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Initialize Network
model = CNN(1,10).to(device)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# Train Netowork
for epoch in range(num_epochs):
    correct = 0
    total = 0
    loop = tqdm(test_loader, desc=f'Epoch: [{epoch+1}/{num_epochs}]')

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

# Train loss = 0.0389, accuracy = 98.32%
# Test accuracy = 98.65%