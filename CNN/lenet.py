import torch
import torch.nn as nn

# LeNet Architecture
# 1x32x32 Input -> (5x5),s=1,p=0 -> avgpool(2x2),s=2,p=0 -> (5x5),s=1,p=0 -> avgpool(2x2),s=2,p=0
# -> Conv 5x5 to 120 channels -> Linear 84 -> Linear 10

class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0)
        self.conv3 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1, padding=0)
        self.fc1 = nn.Linear(120,84)
        self.fc2 = nn.Linear(84,10)

    def forward(self,x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = x.reshape(x.shape[0],-1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x