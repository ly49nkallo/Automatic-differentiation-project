import autograd.numpy as np
from Modules import Linear, HyperbolicTangent, Sigmoid
from Network import Net

# training data
# ig its a XOR gate
x_train = np.array([[[0,0]], [[0,1]], [[1,0]], [[1,1]]])
y_train = np.array([[[0]], [[1]], [[1]], [[0]]])

net = Net()
net.add(Linear(2,3))
net.add(HyperbolicTangent())
net.add(Linear(3,1))
net.add(HyperbolicTangent())

net.train(x_train, y_train, epochs = 1000, lr = 1e-1)

out = net.predict(x_train)
print(np.dot(out, y_train))