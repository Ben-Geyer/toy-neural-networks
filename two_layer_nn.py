import numpy as np

def sigmoid(x, deriv=False):
    if(deriv==True):
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))

def error(lf, y, deriv=False):
    if(deriv==True):
        return 2 * (lf - y)
    return (lf - y) ** 2

#input data
X = np.array([[0, 0, 1],
              [0, 1, 1],
              [1, 0, 1],
              [1, 1, 1]])

#ouput data
y = np.array([[0, 0, 1, 1]]).T

np.random.seed(1)

#set weights
syn0 = 2 * np.random.rand(3,1) - 1

for i in range(10000):
    l0 = X
    l1 = sigmoid(np.dot(l0, syn0))

    l1_error = error(l1, y)

    if (i % 1000 == 0):
        print(l1_error)

    l1_delta = 1 * error(l1, y, True) * sigmoid(l1, True)

    syn0 -= np.dot(l0.T, l1_delta)

print("Output after training: ")
print(l1)

test = np.array([[1, 0, 0],
                 [1, 1, 0],
                 [0, 1, 0],
                 [0, 0, 1]])

print("Weights used: ")
print(syn0)

l0 = test
l1 = sigmoid(np.dot(l0, syn0))

print("Test result: ")
print(l1)
