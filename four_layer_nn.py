import numpy as np

def sigmoid(x, deriv=False):
    if(deriv==True):
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))

def error(lf, y, deriv=False):
    if(deriv==True):
        return 2 * (lf - y)
    return (lf - y) ** 2

def convert_output(o):
    if (o >= .5):
        return 1
    else:
        return 0;


#Layer sizes
i = 8
h0 = 14
h1 = 14
o = 8

#input data
X = np.array([[0, 0, 1, 0, 1, 1, 0, 1],
              [0, 1, 1, 1, 1, 0, 0, 0],
              [1, 0, 1, 0, 0, 0, 1, 1],
              [1, 1, 1, 1, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0],
              [0, 1, 1, 0, 0, 0, 1, 0],
              [0, 1, 1, 0, 1, 0, 1, 1],
              [1, 1, 1, 1, 1, 1, 1, 1]])

#ouput data
y = np.array([[1, 1, 0, 1, 0, 0, 1, 0],
              [1, 0, 0, 0, 0, 1, 1, 1],
              [0, 1, 0, 1, 1, 1, 0, 0],
              [0, 0, 0, 0, 1, 1, 1, 1],
              [1, 1, 1, 1, 1, 1, 1, 1],
              [1, 0, 0, 1, 1, 1, 0, 1],
              [1, 0, 0, 1, 0, 1, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0]])

np.random.seed(1)

#set weights -- experimenting with different network sizes
syn0 = 2 * np.random.rand(i,h0) - 1
syn1 = 2 * np.random.rand(h0,h1) - 1
syn2 = 2 * np.random.rand(h1,o) - 1

#biases
b0 = 2 * np.random.rand(1,h0) - 1
b1 = 2 * np.random.rand(1,h1) - 1
b2 = 2 * np.random.rand(1,o) - 1

for i in range(10000):
    l0 = X
    l1 = sigmoid(np.dot(l0, syn0) + b0)
    l2 = sigmoid(np.dot(l1, syn1) + b1)
    l3 = sigmoid(np.dot(l2, syn2) + b2)

    l3_error = error(l3, y)

    if (i % 1000 == 0):
        print(l3_error)

    l3_delta = 0.1 * error(l3, y, True) * sigmoid(l3, True) #1 x o matrix
    l2_delta = np.dot(l3_delta, syn2.T) * sigmoid(l2, True) #1 x h1
    l1_delta = np.dot(l2_delta, syn1.T) * sigmoid(l1, True) #1 X h0

    syn2 -= np.dot(l2.T, l3_delta)
    syn1 -= np.dot(l1.T, l2_delta)
    syn0 -= np.dot(l0.T, l1_delta)
    
    #Bias by sum of training examples
    b2 -= np.sum(l3_delta, axis = 0)
    b1 -= np.sum(l2_delta, axis = 0)
    b0 -= np.sum(l1_delta, axis = 0)

vout = np.vectorize(convert_output)
l3 = vout(l3)

print("Delta: ")
print(l2_delta)

print("Output after training: ")
print(l3)

test = np.array([[0, 0, 0, 1, 1, 0, 1, 1],
                 [0, 0, 0, 0, 0, 0, 1, 0],
                 [1, 1, 0, 0, 1, 0, 1, 0],
                 [1, 0, 0, 0, 1, 1, 0, 0]])

l0 = test
l1 = sigmoid(np.dot(l0, syn0) + b0)
l2 = sigmoid(np.dot(l1, syn1) + b1)
l3 = sigmoid(np.dot(l2, syn2) + b2)
l3 = vout(l3)

print("Test result: ")
print(l3)
