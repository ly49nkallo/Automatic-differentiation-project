import numpy as np

from autograd import Tensor, Parameter, Module
from autograd.optim import SGD

x_data = Tensor(np.random.randn(100, 3))
coef = Tensor(np.array([-1, +3, -2], dtype=float))
y_data = x_data @ coef + 5

class Model(Module):
    def __init__(self) -> None:
        super().__init__()
        self.w = Parameter(3)
        self.b = Parameter()

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs @ self.w + self.b


batch_size = 32
model = Model()
optimizer = SGD(model.parameters(), lr=0.001)

for epoch in range(100):
    epoch_loss = 0.0

    for start in range(0, 100, batch_size):
        end = start + batch_size

        optimizer.zero_grad()

        inputs = x_data[start:end]

        predicted = model.forward(inputs)
        actual = y_data[start:end]
        errors = predicted - actual
        loss = (errors * errors).sum()

        loss.backward()
        epoch_loss += loss.data

        optimizer.step()

    print(epoch, epoch_loss)