#2020-06-17 new version
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from IPython.display import set_matplotlib_formats
set_matplotlib_formats('png', 'pdf')

#2020-06-17 new
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version=1)

x_org, y_org = mnist.data, mnist.target.astype(np.int)

# step1
x_norm = x_org / 255.0

x_all = np.insert(x_norm, 0, 1, axis=1)

print('x_all shape: ', x_all.shape)

# step2
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse=False)
y_all_one = ohe.fit_transform(np.c_[y_org])
print('After One Hot Vector:', y_all_one.shape)

#step3
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test, y_train_one, y_test_one = train_test_split(
    x_all, y_org, y_all_one, train_size=60000, test_size=10000, shuffle=False
)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape,
      y_train_one.shape, y_test_one.shape)

#2020-06-17
N = 20
np.random.seed(12)
indexes = np.random.choice(y_test.shape[0], N, replace=False)
x_selected = x_test[indexes, 1:]
y_selected = y_test[indexes]
plt.figure(figsize=(10,3))
for i in range(N):
    ax = plt.subplot(2, N/2, i + 1)
    plt.imshow(x_selected[i].reshape(28, 28), cmap='gray_r')
    ax.set_title('%d' % y_selected[i], fontsize=16)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()

def sigmoid(x):
    return 1/(1+ np.exp(-x))

def softmax(x):
    x = x.T
    x_max = x.max(axis=0)
    x = x - x_max
    w = np.exp(x)
    return (w / w.sum(axis=0)).T

def cross_entropy(yt, yp):
    return -np.mean(np.sum(yt * np.log(yp), axis=1))

from sklearn.metrics import accuracy_score

def evaluate(x_test, y_test, y_test_one, V, W):
    b1_test = np.insert(sigmoid(x_test @V), 0, 1, axis=1)
    yp_test_one = softmax(b1_test @ W)
    yp_test = np.argmax(yp_test_one, axis=1)
    loss = cross_entropy(y_test_one, yp_test_one)
    score = accuracy_score(y_test, yp_test)
    return score, loss

import numpy as np

class Indexes():

    def __init__(self, total, size):
        self.total = total
        self.size = size
        self.indexes = np.zeros(0)

    def next_index(self):
        next_flag = False

        if len(self.indexes) < self.size:
            self.indexes = np.random.choice(self.total,
                self.total, replace=False)
            next_flag = True

        index = self.indexes[:self.size]
        self.indexes = self.indexes[self.size:]
        return index, next_flag

indexes = Indexes(20, 5)

for i in range(6):
    arr, flag = indexes.next_index()
    print(arr, flag)

# ReLU
def ReLU(x):
    return np.maximum(0, x)

# step
def step(x):
    return 1.0 * (x > 0)

xx = np.linspace(-4, 4, 501)
yy = ReLU(xx)
plt.figure(figsize=(6,6))
#plt.ylim(0.0, 1.0)
plt.xlim(-4, 4)
plt.ylim(-4, 4)
plt.xlabel(r'$x$', fontsize=14)
plt.ylabel(r'$y$', fontsize=14)
plt.grid(lw=2)
plt.plot(xx, ReLU(xx), c='b', label='ReLU', linestyle='-', lw=3)
plt.plot(xx, step(xx), c='k', label='step', linestyle='-.', lw=3)
plt.xticks(size=14)
plt.yticks(size=14)
plt.legend(fontsize=14)
plt.show()

from sklearn.metrics import accuracy_score

def evaluate3(x_test, y_test, y_test_one, U, V, W):
    b1_test = np.insert(ReLU(x_test @ U), 0, 1, axis=1)
    d1_test = np.insert(ReLU(b1_test @ V), 0, 1, axis=1)
    yp_test_one = softmax(d1_test @ W)
    yp_test = np.argmax(yp_test_one, axis=1)
    loss = cross_entropy(y_test_one, yp_test_one)
    score = accuracy_score(y_test, yp_test)
    return score, loss

#
H = 128
H1 = H + 1

M = x_train.shape[0]
D = x_train.shape[1]

N = y_train_one.shape[1]

alpha = 0.01
nb_epoch = 200
batch_size = 512
B = batch_size

np.random.seed(123)
U = np.random.randn(D, H) / np.sqrt(D / 2)
V = np.random.randn(H1, H) / np.sqrt(H1 / 2)
W = np.random.randn(H1, N) / np.sqrt(H1 / 2)

history4 = np.zeros((0,3))
indexes = Indexes(M, batch_size)
epoch = 0

print(V[:2,:5])
print(W[:2,:5])

while epoch < nb_epoch:
    index, next_flag = indexes.next_index()
    x, yt = x_train[index], y_train_one[index]

    a = x @ U #(10.6.11)
    b = ReLU(a) #(10.6.12)ReLU
    b1 = np.insert(b, 0, 1, axis=1)
    c = b1 @ V     #(10.6.13)
    d = ReLU(c)    #(10.6.14)
    d1 = np.insert(d, 0, 1, axis=1)
    u = d1 @ W      #(10.6.15)
    yp = softmax(u)  #(10.6.16)

    yd = yp - yt #(10.6.17)
    dd = step(c) * (yd @ W[1:].T) #(10.6.18)
    bd = step(a) * (dd @ V[1:].T) #(10.6,19)

    W = W - alpha * (d1.T @ yd) / B #(10.6.20)
    V = V - alpha * (b1.T @ dd) / B #(10.6.21)
    U = U - alpha * (x.T @ bd) / B #(10.6.22)

    if next_flag: #lepoch
        score, loss = evaluate3(
            x_test, y_test, y_test_one, U, V, W
        )
        history4 = np.vstack((history4,
            np.array([epoch, loss, score])))
        print("epoch = %d loss = %f score = %f"
            % (epoch, loss, score))
        epoch = epoch + 1

print('initial state: loss: %f accuracy: %f'
      % (history4[0,1], history4[0,2]))
print('final state: loss: %f accuracy: %f'
      % (history4[-1, 1], history4[-1, 2]))

# loss figure
plt.plot(history4[:,0], history4[:,1])
plt.ylim(0,2.5)
plt.xticks(size=14)
plt.yticks(size=14)
plt.grid(lw=2)
plt.show()

# accuracy figure
plt.plot(history4[:,0], history4[:,2])
plt.ylim(0,1)
plt.xticks(size=14)
plt.yticks(size=14)
plt.grid(lw=2)
plt.show()

#figure 10-22
import matplotlib.pyplot as plt
N = 20
np.random.seed(12)
indexes = np.random.choice(y_test.shape[0], N, replace=False)

x_selected = x_test[indexes]
y_selected = y_test[indexes]

b1_test = np.insert(ReLU(x_selected @ U), 0, 1, axis=1)
d1_test = np.insert(ReLU(b1_test @ V), 0, 1, axis=1)
yp_test_one = softmax(d1_test @ W)
yp_test = np.argmax(yp_test_one, axis=1)

plt.figure(figsize=(10,3))
for i in range(N):
    ax = plt.subplot(2, N/2, i + 1)
    plt.imshow(x_selected[i, 1:].reshape(28, 28), cmap='gray_r')
    ax.set_title('%d:%d' % (y_selected[i], yp_test[i]), fontsize=14)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()


