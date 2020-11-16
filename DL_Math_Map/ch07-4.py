import numpy as np
import matplotlib.pyplot as plt

from IPython.display import set_matplotlib_formats
set_matplotlib_formats('png', 'pdf')

from sklearn.datasets import load_boston
#load data
boston = load_boston()
x_org, yt = boston.data, boston.target
feature_names = boston.feature_names
print('origin data: ', x_org.shape, yt.shape)
print('feature name: ', feature_names)

#RM sources
x_data = x_org[:,feature_names == 'RM']
print('arranged:', x_data.shape)


#insert x_0 = 1
x = np.insert(x_data, 0, 1.0, axis=1)
print('insert x_0: ', x.shape)
# insert LSTAT
x_add = x_org[:, feature_names == 'LSTAT']
x2 = np.hstack((x, x_add))
print(x2.shape)
print(x2[:5, : ])

print(yt[:5])

#distribute figure7-3
plt.scatter(x[:,1], yt, s=10, c='b')
plt.xlabel('ROOM', fontsize=14)
plt.ylabel('PRICE', fontsize=14)
plt.show()

# pred function compute yp
def pred(x, w):
    return (x @ w)

# initial process
M = x2.shape[0]
D = x2.shape[1]
iters = 2000
alpha = 0.001
w = np.ones(D)
history = np.zeros((0,2))

# iterator compute
for k in range(iters):
    yp = pred(x2, w)
    yd = yp - yt
    w = w - alpha * (x2.T @ yd) / M

    if (k % 100 == 0):
        loss = np.mean(yd ** 2) / 2
        history = np.vstack((history, np.array([k, loss])))
        print("iter = %d loss = %f" % (k, loss))

print('Loss initial value: %f' % history[0,1])
print('Loss final value: %f' % history[-1,1])


# learning history
plt.plot(history[1:, 0], history[1:,1])
plt.show()

