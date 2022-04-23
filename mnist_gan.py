from matplotlib import backend_tools
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

from autograd.dataloader import Dataloader
from autograd.tensor import Tensor
import autograd.optim as optim
from autograd.module import Module, Linear
from autograd.activation import *
import autograd.functional as F


class Discriminator(Module):
    def __init__(self, in_features):
        super().__init__()
        self.fc1 = Linear(in_features, 128)
        self.act1 = ReLU()
        self.fc2 = Linear(128, 1)
        self.sig = Sigmoid()
        

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.sig(x)
        return x


class Generator(Module):
    def __init__(self, z_dim, img_dim):
        super().__init__()
        self.fc1 = Linear(z_dim, 256)
        self.act1 = ReLU()
        self.fc2 = Linear(256, img_dim)
        self.tanh = Tanh() # normalize inputs to [-1, 1] so make outputs [-1, 1]

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.tanh(x)
        return x

# https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/GANs/1.%20SimpleGAN/fc_gan.py
# hyperparams
lr = 3e-4
z_dim = 64
image_dim = 28 * 28 * 1  # 784
batch_size = 32
num_epochs = 2

disc = Discriminator(image_dim)
gen = Generator(z_dim, image_dim)
fixed_noise = Tensor(np.random.randn(batch_size, z_dim))
data_loader = Dataloader('mnist', batch_size)
opt_disc = optim.Adam(disc.parameters(), lr=lr)
opt_gen = optim.Adam(gen.parameters(), lr=lr)
criterion = F.nll
step = 0

for epoch in range(num_epochs):
    for batch_idx, (real, _) in enumerate(data_loader):
        real = real.reshape(-1, 784)
        real = Tensor(real)
        batch_size = real.shape[0]

        ### Train Discriminator: max log(D(x)) + log(1 - D(G(z)))  
        noise = Tensor(np.random.randn(batch_size, z_dim))
        fake = gen(noise)
        disc_real = disc(real).view(-1)
        lossD_real = criterion(disc_real, Tensor(np.ones(disc_real.shape)))
        disc_fake = disc(fake).view(-1)
        lossD_fake = criterion(disc_fake, Tensor(np.zeros(disc_fake.shape)))
        lossD = (lossD_real + lossD_fake) / 2
        disc.zero_grad()
        lossD.backward()
        opt_disc.step()

        
        ### Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
        # where the second option of maximizing doesn't suffer from
        # saturating gradients

        output = disc(fake).view(-1)
        lossG = criterion(output, Tensor(np.ones_like(output.data)))
        gen.zero_grad()
        lossG.backward()
        opt_gen.step()

        if batch_idx == 0:
            print(
                f"Epoch [{epoch}/{num_epochs}] Batch {batch_idx}/{len(data_loader)} \
                      Loss D: {lossD:.4f}, loss G: {lossG:.4f}"
            )

plt.show(output)
