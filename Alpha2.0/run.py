from module import Module, Linear, Tanh
from tensor import Parameter, Tensor
from autograd import grad
import numpy as np
class Net(Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = Linear(in_features, out_features)
        self.act = Tanh()
        self.linear2 = Linear(out_features, out_features)
        self.act2 = Tanh()
        self.parame = Parameter([0,0,0,0,0])

    def forward(self, x):
        x = self.linear(x)
        x = self.act(x)
        x = self.linear2(x)
        x = self.act2(x)
        return x

if __name__ == '__main__':
    input = Tensor(np.ones((6,)))
    target = Tensor(np.zeros((5,)))
    model = Net(6, 5)
    print(list(model._parameters))
    print(list(model._modules))
    for idx, m in enumerate(model.named_modules()):
        print(idx, '->', m)
    print('---parameters---')
    for idx, p in enumerate(model.named_parameters()):
        print(idx, '->', p)    


    output:Tensor = model(input)
    print(output, output.data)
    def mse(outputs,target):
        mse = 0
        for o, t in zip(outputs, target):
            mse += 1/5 * (t - o) ** 2
        return -mse
    _grad_fn = grad(mse)
    _grad = _grad_fn(input.data, target.data)
    print(_grad_fn(input.data, target.data))
    print('mse before', mse(output.data, target.data))
    # train
    params = model.parameters()
    print([p for p in params])
    for p in params:
        p.data = [row - _grad for row in p.data]
    print([p for p in params])
    # recalculate 
    print('mse after', mse())
