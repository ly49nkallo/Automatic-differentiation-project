import numpy as np

class base_optimizer():
    def __init__(self, parameters, lr):
        self.parameters = parameters
    def step(self): raise NotImplementedError

class Primitive(base_optimizer):
    def __init__(self, parameters, lr):
        super(Primitive, self).__init__(parameters, lr)
    def step(self):