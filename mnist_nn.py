import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

from autograd.dataloader import Dataloader
from autograd.tensor import Tensor
from autograd.optim import SGD, Momentum, Adam
from autograd.module import Module, Linear
from autograd.activation import *
from autograd.functional import *


class Mlp(Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = Linear(in_features, 128)
        self.linear2 = Linear(128, 32)
        self.linear3 = Linear(32, out_features)
        self.act = Sigmoid()
        self.act2 = Sigmoid()

    def forward(self, x):
        x = self.linear(x)
        x = self.act(x)
        x = self.linear2(x)
        x = self.act2(x)
        x = self.linear3(x)
        return x

def moving_average(a, n=3) :
    if not isinstance(a, np.ndarray): a = np.array(a)
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def main():
    batch_size = 16
    epochs = 1 
    look_back = 10
    def test():
        #model.eval()
        test_loss = 0
        correct = 0
        for data, target in test_loader:
        #data = data.view(-1,28*28)
            data = Tensor(data.reshape((-1, 28*28)), requires_grad = False)
            target = Tensor(target.reshape((-1, 1)))
            output = model(data)
            test_loss += mse(output, target).data
            pred = np.argmax(output.data.copy(), axis=1)
            correct = sum([1 if p == t.data else 0 for p, t in zip(pred, target)])

        test_loss /= len(test_loader.dataset)
        print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(data.data.copy()),
            100. * correct / len(data.data.copy())))
            
    history = []
    loader = Dataloader('mnist', batch_size, shuffle=True, dummy=True)
    test_loader = Dataloader('mnist', 1000, train=False, shuffle=True)
    model = Mlp(28*28, 10)
    optimizer = Adam(model.parameters(), lr = 0.01)
    
    test()
    for i in range(epochs):
        for batch_idx, (data, target) in (enumerate(pbar := tqdm(loader, desc=f"Epoch: {i + 1}", 
                                                    ascii=True, colour='green'))):
            #if (batch_idx == 2000): break
            data = Tensor(data.reshape((-1, 28*28)), requires_grad = True)
            target = Tensor(target.reshape((-1,)))

            optimizer.zero_grad()
            output = model(data)
            loss = nll(output, target)
            pred = np.argmax(output.data.copy(), axis=1)
            correct = sum([1 if p == t.data else 0 for p, t in zip(pred, target)])
            history.append(correct)
            if batch_idx % 100 == 0:
                pbar.set_postfix_str('Accuracy {:.1f}%'.format(sum(history[-look_back:])/look_back/batch_size*100))
            loss.backward()
            optimizer.step()
            

        test()
    #print(history)
   # plt.plot(history)
    plt.plot(moving_average(history[10:], n=10))
    plt.plot(moving_average(history[10:], n=100))
    plt.title('Accuracy')
    plt.show()

if __name__ == '__main__':
    main()