
import matplotlib.pyplot as plt
from autograd.tensor import Tensor

x = Tensor([10, -10, 5, -9, 2, 5], requires_grad=True)
print(x.shape)

history = []
for i in range(10):
    # reset the gradient of x to 0 like
    x.zero_grad()
    sum_of_squares = (x * x).sum()
    sum_of_squares.backward()

    delta_x = Tensor(0.1) * x.grad
    # this strips the gradient from x but we dont need to calculate backwards until next iteration
    x -= delta_x
    
    print(i, sum_of_squares)
    history.append(float(sum_of_squares.data))

plt.plot(history, 'rs')
plt.title('Minimise sum of squares')
plt.show()