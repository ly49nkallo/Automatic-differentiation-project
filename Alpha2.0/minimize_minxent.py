
import matplotlib.pyplot as plt
from autograd.tensor import Tensor
from autograd.functional import minxent

x = Tensor([1, 2, 3, 4, 3, 2, 1], requires_grad=True)
y = Tensor([[1]])
print(x.shape)

history = []
for i in range(1000):
    # reset the gradient of x to 0 like
    x.zero_grad()
    x2 = x.softmax()
    print(x2)
    x2 = minxent(x, y, is_one_hot=False)
    x2.backward()

    delta_x = Tensor(0.5) * x.grad
    # this strips the gradient from x but we dont need to calculate backwards until next iteration
    x -= delta_x
    print(i, x2)
    history.append(float(x2.data))

plt.plot(history, 'rs')
plt.title('Minimise mixent function')
plt.show()