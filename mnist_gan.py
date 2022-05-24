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
        self.act1 = Sigmoid()
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
        assert img_dim == 28 * 28
        super().__init__()
        self.fc1 = Linear(z_dim, 256)
        self.act1 = Sigmoid()
        self.fc2 = Linear(256, img_dim)
        self.tanh = Sigmoid() # normalize inputs to [-1, 1] so make outputs [-1, 1]

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.tanh(x)
        return x

# https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/GANs/1.%20SimpleGAN/fc_gan.py
########### HYPERPARAMS ###########
lr = 3e-4
z_dim = 64
image_dim = 28 * 28 * 1  # 784
batch_size = 32
num_epochs = 2

disc = Discriminator(image_dim)
gen = Generator(z_dim, image_dim)
fixed_noise = Tensor(np.random.randn(batch_size, z_dim))
data_loader = Dataloader('mnist', batch_size, dummy=True)
opt_disc = optim.Adam(disc.parameters(), lr=lr)
opt_gen = optim.Adam(gen.parameters(), lr=lr)
criterion = F.BCELoss
step = 0

for epoch in range(num_epochs):
    for batch_idx, (real, _) in enumerate(data_loader):
        print('begin loop')
        disc.zero_grad()
        gen.zero_grad()
        real = real.reshape(-1, 784)
        real = Tensor(real, requires_grad=True)
        ### Train Discriminator: max log(D(x)) + log(1 - D(G(z)))  
        noise = Tensor(np.random.randn(batch_size, z_dim), requires_grad = True)
        fake = gen(noise)
        disc_real = disc(real)
        lossD_real = criterion(disc_real, Tensor(np.ones(disc_real.shape, dtype=int)))
        disc_fake = disc(fake).view(-1)
        lossD_fake = criterion(disc_fake, Tensor(np.zeros(disc_fake.shape, dtype=int)))
        lossD = (lossD_real + lossD_fake) / 2
        print('lossD:', lossD)
        lossD.backward()
        print('computed discriminator loss')
        opt_disc.step()
        print('steped with disciminator optimizer')
        
        ### Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
        # where the second option of maximizing doesn't suffer from
        # saturating gradients

        output = disc(fake).view(-1)
        assert output.shape == (batch_size,), output.shape
        assert output.requires_grad and output.grad is not None
        print('flattened output of discriminator')
        lossG = criterion(output, Tensor(np.ones_like(output.data)))
        print('lossG:', lossG)
        gen.zero_grad()
        print('zeroed generator gradient')
        print([i.shape for i in list(opt_gen.parameters.__iter__())])
        lossG.backward()
        print('computer generator loss')
        opt_gen.step()

        if batch_idx == 0:
            print(
                f"Epoch [{epoch}/{num_epochs}] Batch {batch_idx}/{len(data_loader)} \
                      Loss D: {lossD:.4f}, loss G: {lossG:.4f}"
            )

plt.show(output)
