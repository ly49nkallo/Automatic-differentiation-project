import autograd.numpy as np
from Modules import Linear, HyperbolicTangent, Sigmoid, ReLU
from Network import Net
from datasets import load_dataset
from autograd import grad

loader = load_dataset('mnist')
x_train, x_test = np.array(loader['train']['image']).astype('float32'), np.array(loader['test']['image']).astype('float32')
y_train, y_test = np.array(loader['train']['label']).astype('float32'), np.array(loader['test']['label']).astype('float32')

print(x_train.shape, y_train.shape, x_test.shape, y_train.shape)
print(x_train.dtype)
x_train=x_train.reshape(x_train.shape[0], 1, 28*28)
print(x_train.shape, y_train.shape, x_test.shape, y_train.shape)
x_train /= 255
x_train.astype('float32')

new_y_train = []
for label in y_train:
    a = [0] * 10
    a[int(label)] = 1
    new_y_train.append(a)
y_train = np.array(new_y_train).astype('float32')

x_test=x_test.reshape(x_test.shape[0], 1, 28*28)

x_test /= 255
x_test.astype('float32')

new_y_test = []
for label in y_test:
    a = [0] * 10
    a[int(label)] = 1
    new_y_test.append(a)
y_test = np.array(new_y_test).astype('float32')

print(x_train.shape, y_train.shape, x_test.shape, y_train.shape)

net = Net()
net.add(Linear(28*28,128))
net.add(Sigmoid())
net.add(Linear(128,128))
net.add(Sigmoid())
net.add(Linear(128,10))
net.add(Sigmoid())

net.train(x_train[0:4000], y_train[0:4000], epochs=9, lr=0.1)
net.train(x_train[4000:8000], y_train[4000:8000], epochs=7, lr=0.001)

TEST_SIZE = 8000
out = net.predict(x_test[:TEST_SIZE])
correct = 0
for i in range(TEST_SIZE):
    response = np.argmax(out[i])
    if response == np.argmax(y_test[i]):
        correct = correct + 1

print(f'Accuracy is {correct/TEST_SIZE*100}%')