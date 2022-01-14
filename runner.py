from utils import Dataloader

loader = Dataloader('MNIST', 32, train=True)

print(iter(loader).__next__().shape)