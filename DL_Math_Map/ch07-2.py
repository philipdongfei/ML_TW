import numpy as np
import matplotlib.pyplot as plt

from IPython.display import set_matplotlib_formats
set_matplotlib_formats('png', 'pdf')

#w=(1,2)
w = np.array([1,2])
print(w)
print(w.shape)

#x = (3,4)
x = np.array([3,4])
print(x)
print(x.shape)

#y = 1*3 + 2*4 = 11
y = x @ w
print(y)


# X is 3X2 matrix
X = np.array([[1,2],[3,4],[5,6]])
print(X)
print(X.shape)
Y = X @ w
print(Y)
print(Y.shape)

XT = X.T
print("X:\n", X)
print("XT:\n", XT)

yd = np.array([1,2,3])
print("yd:\n", yd)

grad = XT @ yd
print(grad)

loss = np.mean(yd**2)/2
print('loss:', loss)
