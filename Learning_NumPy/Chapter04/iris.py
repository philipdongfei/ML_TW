import numpy as np
import pandas as pd

df = pd.read_csv('iris.data', header=None)
x = df.iloc[0:100,[0,1,2,3]].values
y = df.iloc[0:100,4].values
y = np.where(y=='Iris-setosa',0,1)

x_train = np.empty((80,4))
x_test = np.empty((20,4))
y_train = np.empty(80)
y_test = np.empty(20)

x_train[:40],x_train[40:] = x[:40],x[50:90]
x_test[:10],x_test[10:] = x[40:50],x[90:100]
y_train[:40],y_train[40:] = y[:40],y[50:90]
y_test[:10],y_test[10:] = y[40:50],y[90:100]

def sigmoid(x):
    return 1/(1+np.exp(-x))

def activation(x, w, b):
    return sigmoid(np.dot(x, w) + b)

def update(x, y, w, b, eta):
    y_pred = activation(x, w, b)
    a = (y_pred - y_train) * y_pred * (1 - y_pred)
    for i in range(4):
        w[i] -= eta * 1/float(len(y)) * np.sum(a*x[:,i])
    b -= eta * 1/float(len(y))*np.sum(a)
    return w, b

weights = np.ones(4)/10
bias = np.ones(1)/10
for _ in range(999):
    weights, bias = update(x_train, y_train, weights, bias, eta=0.1)
print('weights = ', weights, 'bias = ', bias)

print(activation(x_test, weights, bias))


