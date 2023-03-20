import autograd as a
import autograd.utils as u
import numpy as np
'''INVESTIGATE CRITICALITY IN MULTI-LAYER PERCEPTRONS'''

class MLP(a.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = a.Linear(100,100)
        self.linear2 = a.Linear(100,100)
        self.act1 = a.Sigmoid()
        self.act2 = a.Sigmoid()

    def forward(self, x):
        x = self.linear1(x)
        x = self.act1(x)
        x = self.linear2(x)
        x = self.act2(x)
        return x
    
def main():
    dataloader = a.Dataloader(dataset='NOT', batch_size=10, train=False)
    test_dataloader = a.Dataloader(dataset='NOT', batch_size=10, train=True)
    history = []
    model = MLP()
    optim = a.SGD(model.parameters(), lr=0.02)
    u.serialize_model(model)
    del model
    model = u.load_model(name='MLP')
    print(list(model.named_parameters()))


if __name__ == '__main__':
    main()