# Make distributions Pg and Pr similar 
# Pg is from generator and Pr is for real images
# Use Wasserstein distance 
# Generate separates as much as possible, Discriminator was the minimize these

import torch
import torch.nn as nn
import torch.optim  as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# Models
class Discriminator(nn.Module):
    def __init__(self, in_channels, features):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=5, stride=2, padding=2, bias=False),
            nn.LeakyReLU(0.2),
            self._block(features, features*2, 4, 2, 1),
            self._block(features*2, features*4, 4, 2, 1),
            self._block(features*4, features*8, 4, 2, 1),
            nn.Conv2d(features*8, 1, 4, 1, 0)
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2) 
        )

    def forward(self,x):
        return self.disc(x)
    
class Generator(nn.Module):
    def __init__(self, z_dims, features, out_channels):
        super().__init__()
        self.gen = nn.Sequential(
            self._block(z_dims, features*8, 4, 1, 0),
            self._block(features*8, features*4, 4, 2, 1),
            self._block(features*4, features*2, 4, 2, 1),
            self._block(features*2, features, 4, 2, 1),
            nn.ConvTranspose2d(features, out_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    
    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU() 
        )

    def forward(self,x):
        return self.gen(x)
    
# Initalize weights with mean = 0, std 0.02
def init_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
    
# Hyper parameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lr = 5e-5
z_dims = 100
img_channels = 1
features = 32
batch_size = 64
num_epochs = 10
img_size = 64
critic = 5
clip = 0.01

disc = Discriminator(img_channels, features).to(device)
init_weights(disc)
gen = Generator(z_dims, features, img_channels).to(device)
init_weights(gen)

fixed_noise = torch.randn((32, z_dims, 1, 1)).to(device)
transform = transforms.Compose([
    transforms.Resize(img_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*img_channels, [0.5]*img_channels)
])

dataset = datasets.MNIST(root='../../Datasets/', transform=transform, download=False)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

opt_disc = optim.RMSprop(disc.parameters(), lr=lr)
opt_gen = optim.RMSprop(gen.parameters(), lr=lr)

writer_real = SummaryWriter(f'runs/real')
writer_fake = SummaryWriter(f'runs/fake')
step = 0

disc.train()
gen.train()

# Training loop
for epoch in range(num_epochs):
    loop = tqdm(loader, desc=f'Epoch: [{epoch+1}/{num_epochs}]')
    idx = 0

    for real,_ in loop:
        real = real.to(device)

        # Train discriminator
        for _ in range(critic):
            noise = torch.randn(real.shape[0], z_dims, 1, 1).to(device)
            fake = gen(noise)
            disc_real = disc(real).view(-1)
            disc_fake = disc(fake).view(-1)
            loss_disc = -(torch.mean(disc_real) - torch.mean(disc_fake)) # Negative of minimizing is maximizing

            disc.zero_grad()
            loss_disc.backward()
            opt_disc.step()

            for p in disc.parameters():
                p.data.clamp_(-clip, clip)

        # Train Generator
        noise = torch.randn(real.shape[0], z_dims, 1, 1).to(device)
        fake = gen(noise)
        output = disc(fake).view(-1)
        loss_gen = -torch.mean(output)
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        if idx % 250 == 0:
            with torch.no_grad():
                fake = gen(fixed_noise)
                img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
                img_grid_real = torchvision.utils.make_grid(real, normalize=True)

                writer_fake.add_image("Fake", img_grid_fake, global_step=step)
                writer_real.add_image("Real", img_grid_real, global_step=step)
                step += 1
                torchvision.utils.save_image(img_grid_fake, f'fake_images/epoch{step}.png')
        idx+=1