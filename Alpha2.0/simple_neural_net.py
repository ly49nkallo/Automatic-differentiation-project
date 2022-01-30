import numpy as np
import matplotlib.pyplot as plt

from autograd import Tensor, Module
from autograd.optim import SGD
from autograd.module import Linear, Sigmoid

def xor_gate(a, b):
    assert isinstance(a, int) and isinstance(b, int)
    if a != b:
        return 1
    else:
        return 0

class model(Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = Linear(in_features, 5)
        self.linear2 = Linear(5, out_features)
        self.act = Sigmoid()
        self.act2 = Sigmoid()

    def forward(self, x):
        x = self.linear(x)
        x = self.act(x)
        x = self.linear2(x)
        x = self.act2(x)
        return x

def mse(output, label):
    return (label - output) ** 2

if __name__ == '__main__':
    pairs = [[np.random.randint(0,2) for _ in range(2)] for i in range(1000)]
    labels = [[xor_gate(a,b)] for a,b in pairs]
    x_train = Tensor(pairs)
    y_train = Tensor(labels)

    net = model(2, 1)
    print(list(net.parameters()))
    optimizer = SGD(net.parameters(), lr=0.01)
    batch_size = 32
    print(x_train.shape, y_train.shape)

    history = []
    starts = np.arange(0, x_train.shape[0], batch_size)
    for epoch in range(200):
        epoch_loss = 0.0

        np.random.shuffle(starts)
        for start in starts:
            end = start + batch_size

            optimizer.zero_grad()

            inputs = x_train[start:end]

            predicted = net(inputs)
            actual = y_train[start:end]
            errors = predicted - actual
            loss = (errors * errors).sum()

            loss.backward()
            epoch_loss += loss.data

            optimizer.step()
            
        history.append(epoch_loss)
        #print(epoch, epoch_loss)
    
    plt.plot(history[10:])
    plt.title('loss')
    plt.show()
