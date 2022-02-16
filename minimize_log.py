
import matplotlib.pyplot as plt
from autograd.tensor import Tensor

x = Tensor([1, 2, 3, 4, 3, 2, 1], requires_grad=True)
print(x.shape)

history = []
for i in range(500):
    # reset the gradient of x to 0 like
    x.zero_grad()
    x2 = x.log().sum()
    x2.backward()

    delta_x = Tensor(0.001) * x.grad
    # this strips the gradient from x but we dont need to calculate backwards until next iteration
    x -= delta_x
    print(i, x2)
    history.append(float(x2.data))

plt.plot(history, 'rs')
plt.title('Minimise exponential function')
plt.show()