from utils import Dataloader

if __name__ == '__main__':
    loader = Dataloader('MNIST', 32, train=True)

    for i, (imgs, labels) in enumerate(loader):
        print(i, imgs.shape, labels.shape)
    
        

