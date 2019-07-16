import numpy as np


def init_network():
    network = {}
    network['W1'] = np.array([[-2.0, -2.0], [3.0, 1.0]])
    network['b1'] = np.array([-1.0, 1.0])
    network['W2'] = np.array([[-60.0], [94.0]])
    network['b2'] = np.array([15.0])

    return network


def sigmoid(r):
    r = 1/(1+np.exp(r))
    return r

def forward(network, x):
    W1, W2 = network['W1'], network['W2']
    b1, b2= network['b1'], network['b2']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2

    y = sigmoid(a2)

    return y


network = init_network()
x = np.array([1.0, 1.0])
y = forward(network, x)
print(y)