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
    def __init__(self, in_features):
        super().__init__()
        self.disc = nn.Sequential(
            nn.Linear(in_features, 100),
            nn.LeakyReLU(0.1),
            nn.Linear(100,1),
            nn.Sigmoid()
        )

    def forward(self,x):
        return self.disc(x)
    
class Generator(nn.Module):
    def __init__(self, z_dims, out_features):
        super().__init__()
        self.gen = nn.Sequential(
            nn.Linear(z_dims, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256,out_features),
            nn.Tanh()
        )

    def forward(self,x):
        return self.gen(x)
    
# Hyper parameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
lr = 3e-4
z_dims = 100
img_dims = 28*28*1
batch_size = 32
num_epochs = 50

disc = Discriminator(img_dims).to(device)
gen = Generator(z_dims, img_dims).to(device)
fixed_noise = torch.rand((batch_size, z_dims)).to(device)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

dataset = datasets.MNIST(root='../../Datasets/', transform=transforms.ToTensor(), download=False)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

opt_disc = optim.Adam(disc.parameters(), lr=lr)
opt_gen = optim.Adam(gen.parameters(), lr=lr)
criterion = nn.BCELoss()

writer_real = SummaryWriter(f'runs/real')
writer_fake = SummaryWriter(f'runs/fake')
step = 0

# Training loop
for epoch in range(num_epochs):
    loop = tqdm(loader, desc=f'Epoch: [{epoch+1}/{num_epochs}]')
    idx = 1
    total_loss_disc = 0
    total_loss_gen = 0

    for real,_ in loop:
        real = torch.flatten(real,1).to(device)

        # Train discriminator
        noise = torch.randn(real.shape[0], z_dims).to(device)
        fake = gen(noise)
        out_real = disc(real).view(-1)
        loss_real = criterion(out_real, torch.ones_like(out_real))
        out_fake = disc(fake).view(-1)
        loss_fake = criterion(out_fake, torch.zeros_like(out_fake))
        loss_disc = (loss_fake + loss_real) / 2
        total_loss_disc += loss_disc.item()

        disc.zero_grad()
        loss_disc.backward(retain_graph = True)
        opt_disc.step()

        # Train Generator
        output = disc(fake).view(-1)
        loss_gen = criterion(output, torch.ones_like(output))
        total_loss_gen += loss_gen.item()
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        if idx % 10 == 0: loop.set_postfix(lossD = f'{(total_loss_disc / idx):.4f}', lossG = f'{(total_loss_gen / idx):.4f}')
        idx+=1

        if idx == 10:
            with torch.no_grad():
                fake = gen(fixed_noise).reshape(-1,1,28,28)
                real = real.reshape(-1,1,28,28)
                img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)
                img_grid_real = torchvision.utils.make_grid(real[:32], normalize=True)

                writer_fake.add_image("Fake", img_grid_fake, global_step=step)
                writer_real.add_image("Real", img_grid_real, global_step=step)
                step += 1

                torchvision.utils.save_image(img_grid_fake, f'fake_images/epoch{epoch+1}.png')