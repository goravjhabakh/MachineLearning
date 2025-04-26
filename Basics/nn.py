# Imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tqdm import tqdm

# Create Fully Connected Neural Network
class NN(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(input_size,100)
        self.fc2 = nn.Linear(100,num_classes)

    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
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
model = NN(784,10).to(device)

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
        data = data.reshape(data.shape[0], -1)

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
        x = x.reshape(x.shape[0], -1)

        output = model(x)
        _, pred = output.max(1)  # val,ind
        correct += (pred == y).sum()
        total += pred.size(0)
        acc = correct / total * 100
    print(f'Accuracy: {acc:.2f}%')

# Train loss = 0.0983, accuracy = 97.73%
# Test accuracy = 98.32%