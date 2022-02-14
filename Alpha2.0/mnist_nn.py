import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from tqdm import tqdm
import time

from autograd.tensor import Tensor
from autograd.optim import SGD
from autograd.module import Module, Linear
from autograd.activation import Sigmoid, Tanh, Softmax
from autograd.functional import *

def moving_average(a, n=3) :
    if not isinstance(a, np.ndarray): a = np.array(a)
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def main():
    def test():
        #model.eval()
        test_loss = 0
        correct = 0
        for data, target in test_loader:
        #data = data.view(-1,28*28)
            data = Tensor(data.reshape((-1, 28*28)), requires_grad = False)
            target = Tensor(target.reshape((-1, 1)))
            print('test', data.shape, target.shape)
            output = model(data)
            test_loss += mse(output, target).data
            print(output.shape, target.shape)
            pred = np.argmax(output.data.copy(), axis=1)
            print(pred.shape, target.shape)
            correct = sum([1 if p == t.data else 0 for p, t in zip(pred, target)])

            break
        test_loss /= len(test_loader.dataset)
        print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(data.data.copy()),
            100. * correct / len(data.data.copy())))
            
    history = []
    loader = Dataloader('mnist', 16)
    test_loader = Dataloader('mnist', 1000, train=False)
    model = Mlp(28*28, 10)
    optimizer = SGD(model.parameters(), lr = 0.01)
    epochs = 1
    test()
    time.sleep(1)
    for i in range(epochs):
        for batch_idx, (data, target) in (enumerate(tqdm(loader, desc=f"Epoch: {i + 1}", ascii=True, colour='green'))):
            data = Tensor(data.reshape((-1, 28*28)), requires_grad = True)
            target = Tensor(target.reshape((-1,)))
            #print(data.shape, target.shape)
            #print(batch_idx, data.shape, target)
            optimizer.zero_grad()
            output = model(data)
            #assert output.shape == (16, 10), output.shape
            #assert target.shape == (16,), target.shape
            loss = nll(output, target)
            history.append(loss.data.copy())
            loss.backward()
            optimizer.step()
            #print(batch_idx, 'loss', loss.data)
            if batch_idx >= (60000//16): 
                print('ending stats')
                print(output.data.shape)
                print(target.data.shape)
                print('final output data:', output.data)
                print('target data:', target.data)
                break
        print('Epoch', i+1)
        test()
   # plt.plot(history)
    plt.plot(moving_average(history[10:], n=10))
    plt.plot(moving_average(history[10:], n=100))
    plt.show()
'''Author: Ty Brennan'''

class Dataloader:
    '''A class that allows us to get batches of data from huggingface datsets'''
    def __init__(self, dataset:str,batch_size, transforms:list=list(), train=True):
        self.dataset = dataset.upper()
        self.batch_size = batch_size
        self.train = train
        assert isinstance(self.dataset, str)
        if self.dataset == 'MNIST':
            if train:
                self.pairs = load_dataset('mnist', split='train')
            else:
                self.pairs = load_dataset('mnist', split='test')

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index < len(self.pairs):
            batch = self.pairs[self.index:self.index+self.batch_size]
            self.index += self.batch_size
            # we want to return a tuple of two npArrays of shape 32x28x28 and 32x1
            # getdata() has been depreciated
            return np.array([np.array(b) / 255 for b in batch['image']]).reshape(-1, 28, 28), np.array(batch['label'])
        else:
            raise StopIteration

    def __len__(self):
        return len(self.pairs) // self.batch_size
    

class Mlp(Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = Linear(in_features, 128)
        #self.linear2 = Linear(128, 128)
        self.linear3 = Linear(128, out_features)
        self.act = Sigmoid()
        #self.act2 = Sigmoid()
        self.softmax = Softmax()

    def forward(self, x):
        x = self.linear(x)
        x = self.act(x)
        #x = self.linear2(x)
        #x = self.act2(x)
        x = self.linear3(x)
        #x = self.act2(x)
        #x = self.softmax(x)
        return x


if __name__ == '__main__':
    main()