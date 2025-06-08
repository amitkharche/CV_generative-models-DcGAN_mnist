import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from src.models.generator import Generator
from src.models.discriminator import Discriminator
import os
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
noise_dim = 100
epochs = 30
batch_size = 128
sample_dir = "output/samples"
os.makedirs(sample_dir, exist_ok=True)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

dataset = datasets.MNIST(root='data', train=True, transform=transform, download=True)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

generator = Generator(noise_dim).to(device)
discriminator = Discriminator().to(device)

criterion = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

fixed_noise = torch.randn(64, noise_dim).to(device)

def save_generated_images(epoch):
    generator.eval()
    with torch.no_grad():
        fake_images = generator(fixed_noise).cpu()
    grid = fake_images.view(64, 1, 28, 28)
    grid = grid.permute(0, 2, 3, 1).squeeze()
    fig, axes = plt.subplots(8, 8, figsize=(8, 8))
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(grid[i], cmap='gray')
        ax.axis('off')
    plt.tight_layout()
    plt.savefig(f"{sample_dir}/epoch_{epoch:03d}.png")
    plt.close()

for epoch in range(1, epochs + 1):
    generator.train()
    for real_imgs, _ in dataloader:
        real_imgs = real_imgs.to(device)
        valid = torch.ones(real_imgs.size(0), 1, device=device)
        fake = torch.zeros(real_imgs.size(0), 1, device=device)

        z = torch.randn(real_imgs.size(0), noise_dim).to(device)
        fake_imgs = generator(z)

        optimizer_D.zero_grad()
        real_loss = criterion(discriminator(real_imgs), valid)
        fake_loss = criterion(discriminator(fake_imgs.detach()), fake)
        d_loss = real_loss + fake_loss
        d_loss.backward()
        optimizer_D.step()

        optimizer_G.zero_grad()
        g_loss = criterion(discriminator(fake_imgs), valid)
        g_loss.backward()
        optimizer_G.step()

    print(f"Epoch {epoch}/{epochs} | D_loss: {d_loss.item():.4f} | G_loss: {g_loss.item():.4f}")
    save_generated_images(epoch)

torch.save(generator.state_dict(), "output/generator.pth")
torch.save(discriminator.state_dict(), "output/discriminator.pth")
