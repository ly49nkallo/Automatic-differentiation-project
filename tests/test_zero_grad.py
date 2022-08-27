from autograd.module import Module, Linear
from autograd.parameter import Parameter
from autograd.optim import SGD
from autograd.tensor import Tensor

class MyModule(Module):
    def __init__(self):
        super().__init__()
        self.param_1 = Parameter(5,5)
        self.param_3 = Parameter(5)
        self.moddy = Linear(5,3)
    
    def forward(self, x):
        out = x @ self.param_1 + self.param_3
        return out

def test_zero_grad():
    mod = MyModule()
    print(list(mod.named_modules()))
    for parameter in mod.parameters():
        assert parameter.grad.data.max() == 0
    optim = SGD(mod.parameters())
    output = mod(Tensor([1, 1, 1, 1, 1], requires_grad=True))
    output = (output / 2).sum()
    print(output)
    mod.zero_grad()
    output.backward()
    optim.step()
    output = mod(Tensor([1, 1, 1, 1, 1], requires_grad=True))
    output = (output / 2).sum()
    mod.zero_grad()
    output.backward()
    optim.step()
    
