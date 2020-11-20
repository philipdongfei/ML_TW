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

#
H = 128
H1 = H + 1
M = x_train.shape[0]
D = x_train.shape[1]
N = y_train_one.shape[1]

nb_epoch = 100
batch_size = 512
B = batch_size
alpha = 0.01

V = np.ones((D, H))
W = np.ones((H1, N))

history1 = np.zeros((0, 3))
indexes = Indexes(M, batch_size)

epoch = 0

while epoch < nb_epoch:
    index, next_flag = indexes.next_index()
    x, yt = x_train[index], y_train_one[index]

    a = x @ V    #(10.6.3)
    b = sigmoid(a)   #(10.6.4)
    b1 = np.insert(b, 0, 1, axis=1) #
    u = b1 @ W   # (10.6.5)
    yp = softmax(u) #(10.6.6)

    yd = yp - yt  #(10.6.7)
    bd = b * (1-b) * (yd @ W[1:].T) #(10.6.8)

    #
    W = W - alpha * (b1.T @ yd) / B # (10.6.9)
    V = V - alpha * (x.T @ bd) / B #(10.6.10)

    if next_flag:
        score, loss = evaluate(
            x_test, y_test, y_test_one, V, W
        )
        history1 = np.vstack((history1,
            np.array([epoch, loss, score])))
        print("epoch = %d loss = %f score = %f"
              % (epoch, loss, score))
        epoch = epoch + 1

print('initial state: loss: %f accuracy: %f'
      % (history1[0,1], history1[0,2]))
print('final state: loss: %f accuracy: %f'
      % (history1[-1, 1], history1[-1, 2]))

# loss figure
plt.plot(history1[:,0], history1[:,1])
plt.ylim(0,2.5)
plt.xticks(size=14)
plt.yticks(size=14)
plt.grid(lw=2)
plt.show()

# accuracy figure
plt.plot(history1[:,0], history1[:,2])
plt.ylim(0,1)
plt.xticks(size=14)
plt.yticks(size=14)
plt.grid(lw=2)
plt.show()


