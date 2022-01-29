
import matplotlib.pyplot as plt
from autograd.tensor import Tensor, tensor_sum

x = Tensor([10, -10, 5, -9, 2, 5], requires_grad=True)
print(x.shape)

history = []
for i in range(10):
    x.zero_grad()
    sum_of_squares = tensor_sum(x * x)
    sum_of_squares.backward()

    delta_x = Tensor(0.1) * x.grad
    x -= delta_x
    
    print(i, sum_of_squares)
    history.append(float(sum_of_squares.data))

plt.plot(history)
plt.title('Minimise sum of squares')
plt.show()