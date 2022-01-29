
import matplotlib.pyplot as plt
from autograd.tensor import Tensor, tensor_sum, multiply

x = Tensor([10, -10, 5, -9, 2, 5], requires_grad=True)
print(x.shape)

history = []
for i in range(10):
    sum_of_squares = tensor_sum(multiply(x, x))
    sum_of_squares.backward()

    delta_x = multiply(Tensor(0.1), x.grad)
    x = Tensor(x.data - delta_x.data, requires_grad=True)
    
    print(i, sum_of_squares)
    history.append((i, float(sum_of_squares.data)))

plt.plot(history)
plt.show()