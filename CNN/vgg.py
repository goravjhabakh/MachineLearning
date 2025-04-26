import torch
import torch.nn as nn

# Using D part from paper with only 3x3 conv
vgg_types = {
    11: [64,'M',128,'M',256,256,'M',512,512,'M',512,512,'M'],
    13: [64,64,'M',128,128,'M',256,256,'M',512,512,'M',512,512,'M'],
    16: [64,64,'M',128,128,'M',256,256,256,'M',512,512,512,'M',512,512,512,'M'],
    19: [64,64,'M',128,128,'M',256,256,256,256,'M',512,512,512,512,'M',512,512,512,512,'M']
}

class VGGNet(nn.Module):
    def __init__(self, in_channels = 3, num_classes = 10):
        super().__init__()
        self.in_channels = in_channels
        self.conv = self.generate_conv_layers(vgg_types[16])
        self.fcs = nn.Sequential(
            nn.Linear(512*7*7, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x,1)
        x = self.fcs(x)
        return x

    def generate_conv_layers(self, arch):
        layers = []
        in_channels = self.in_channels

        for x in arch:
            if type(x) == int:
                out_channels = x
                layers += [
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU()
                ]
                in_channels = x
            elif x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]

        return nn.Sequential(*layers)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = VGGNet().to(device)
x = torch.randn(1,3,224,224).to(device)
y = model(x)
print(y.shape)