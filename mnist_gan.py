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
        super().__init__()
        self.fc1 = Linear(z_dim, 256)
        self.act1 = Sigmoid()
        self.fc2 = Linear(256, img_dim)
        self.tanh = Tanh() # normalize inputs to [-1, 1] so make outputs [-1, 1]

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.tanh(x)
        return x

# https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/GANs/1.%20SimpleGAN/fc_gan.py
# STOLEN! in the caribbean!

########### HYPERPARAMS ###########
lr = 3e-4
z_dim = 64
image_dim = 28 * 28 * 1  # 784
batch_size = 32
num_epochs = 2

disc = Discriminator(image_dim)
gen = Generator(z_dim, image_dim)
fixed_noise = Tensor(np.random.randn(batch_size, z_dim), requires_grad = True)
data_loader = Dataloader('mnist', batch_size, dummy=False)
opt_disc = optim.Adam(disc.parameters(), lr=lr)
opt_gen = optim.Adam(gen.parameters(), lr=lr)
criterion = F.BCELoss
step = 0

for epoch in range(num_epochs):
    for batch_idx, (real, _) in (enumerate(tqdm(data_loader, desc=f"Epoch: {epoch + 1}", 
                                                    ascii=True, colour='green'))):
        real = real.reshape(-1, 784)
        real = Tensor(real, requires_grad=True)

        ### Train Discriminator: max log(D(x)) + log(1 - D(G(z))) ###
        noise = Tensor(np.random.randn(batch_size, z_dim), requires_grad = True)
        fake = gen(noise)
        disc_real = disc(real)
        lossD_real = criterion(disc_real, Tensor(np.ones(disc_real.shape, dtype=int)))
        disc_fake = disc(fake).view(-1)
        lossD_fake = criterion(disc_fake, Tensor(np.zeros(disc_fake.shape, dtype=int)))
        lossD = (lossD_real + lossD_fake) / 2
        lossD.backward() #retain graph = false
        opt_disc.step() # alters parameters of the discriminator
        
        ### Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z)) ###
        # where the second option of maximizing doesn't suffer from
        # saturating gradients
        output = disc(fake).view(-1)
        lossG = criterion(output, Tensor(np.ones_like(output.data)))
        disc.zero_grad()
        gen.zero_grad()
        lossG.backward() # error is thrown here because the gradient of lossG depends on the output of disc, which requires math with the disc parameters
        opt_gen.step()
        
        if batch_idx == 0:
            print(
                f"Epoch [{epoch}/{num_epochs}] Batch {batch_idx}/{len(data_loader)} \
                      Loss D: {lossD.data:.4f}, loss G: {lossG.data:.4f}"
            )
fake_cpy = fake.data.copy().reshape(-1, 1, 28, 28)
actual = real.data.copy().reshape(-1, 1, 28, 28)
fig = plt.figure(figsize=(10,7))
rows = 2
columns = 8
i = 0
fake_cpy = fake.data.copy().reshape(-1, 1, 28, 28)
actual = real.data.copy().reshape(-1, 1, 28, 28)
fig = plt.figure(figsize=(10,7))
rows = 2
columns = 8
for i in range(8):
    i += 1
    fig.add_subplot(rows, columns, i)
    plt.imshow(fake_cpy[i-1,0])
    plt.axis('off')
for i in range(8):
    i += 1
    fig.add_subplot(rows, columns, i+8)
    plt.imshow(actual[i-1,0])
    plt.axis('off')
plt.show()

