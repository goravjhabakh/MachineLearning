# Imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tqdm import tqdm

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
input_size = 28
sequence_length = 28
num_layers = 2
hidden_size = 256
num_classes = 10
lr = 0.001
batch_size = 64
num_epochs = 5

# Create Recurrent Neural Network
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size * sequence_length, num_classes)

    def forward(self,x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        #Forward
        out,_ = self.rnn(x,h0)
        out = out.reshape(out.shape[0],-1)
        out = self.fc(out)
        return out
    
# Create GRU
class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size * sequence_length, num_classes)

    def forward(self,x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        #Forward
        out,_ = self.gru(x,h0)
        out = out.reshape(out.shape[0],-1)
        out = self.fc(out)
        return out
    
# Create LSTM
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self,x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(device)

        #Forward
        out,_ = self.lstm(x,(h0,c0))
        out = self.fc(out[:,-1,:])
        return out

# Create Bi-LSTM
class BLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, num_classes)

    def forward(self,x):
        h0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers*2, x.size(0), self.hidden_size).to(device)

        #Forward
        out,_ = self.lstm(x,(h0,c0))
        out = self.fc(out[:,-1,:])
        return out

# Load Data
train_dataset = datasets.MNIST(root='../Datasets/', train=True, transform=transforms.ToTensor(), download=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

test_dataset = datasets.MNIST(root='../Datasets/', train=False, transform=transforms.ToTensor(), download=True)
test_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Initialize Network
#model = RNN(input_size,hidden_size,num_layers,num_classes).to(device)
#model = GRU(input_size,hidden_size,num_layers,num_classes).to(device)
#model = LSTM(input_size,hidden_size,num_layers,num_classes).to(device)
model = BLSTM(input_size,hidden_size,num_layers,num_classes).to(device)

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# Train Netowork
for epoch in range(num_epochs):
    correct = 0
    total = 0
    loop = tqdm(test_loader, desc=f'Epoch: [{epoch+1}/{num_epochs}]')

    for data,target in loop:
        # Forward
        data, target = data.to(device), target.to(device)
        data = data.squeeze(1)
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
        x = x.squeeze(1)
        output = model(x)

        _, pred = output.max(1)  # val,ind
        correct += (pred == y).sum()
        total += pred.size(0)
        acc = correct / total * 100
    print(f'Accuracy: {acc:.2f}%')

# RNN
# Train loss = 0.0335, accuracy = 97.26%
# Test accuracy = 97.97%
#
# GRU
# Train loss = 0.0477, accuracy = 99.07%
# Test accuracy = 99.17%
#
# LSTM
# Train loss = 0.0037, accuracy = 99.09%
# Test accuracy = 99.10%
#
# LSTM (2nd)
# Train loss = 0.04, accuracy = 98.84%
# Test accuracy = 98.96%
#
# Bi-LSTM
# Train loss = 0.2893, accuracy = 98.39%
# Test accuracy = 98.80%