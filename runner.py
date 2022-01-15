from utils import Dataloader

loader = Dataloader('MNIST', 32, train=True)

for i, (imgs, labels) in enumerate(loader):
    print(i, imgs.shape, labels.shape)