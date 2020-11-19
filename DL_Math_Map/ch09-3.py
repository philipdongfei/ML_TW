import numpy as np
import matplotlib.pyplot as plt

from IPython.display import set_matplotlib_formats
set_matplotlib_formats('png', 'pdf')

#input 4D data
from sklearn.datasets import load_iris
iris = load_iris()
x_org, y_org = iris.data, iris.target

x_select = x_org[:,[0,2]]
print('source data shape:', x_select.shape, y_org.shape)

# distribution figure
x_t0 = x_select[y_org == 0]
x_t1 = x_select[y_org == 1]
x_t2 = x_select[y_org == 2]
plt.figure(figsize=(6,6))
plt.scatter(x_t0[:,0], x_t0[:,1], marker='x', c='k', s=50, label='0 (setosa)')
plt.scatter(x_t1[:,0], x_t1[:,1], marker='o', c='b', s=50, label='1 (versicolor)')
plt.scatter(x_t2[:,0], x_t2[:,1], marker='+', c='r', s=50, label='2 (virginica)')
plt.xlabel('sepal_length', fontsize=14)
plt.ylabel('petal_length', fontsize=14)
plt.xticks(size=14)
plt.yticks(size=14)
plt.legend(fontsize=14)
plt.show()

x_all = np.insert(x_select, 0, 1.0, axis=1)



from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse=False, categories='auto')
y_work = np.c_[y_org]
y_all_one = ohe.fit_transform(y_work)
print('y shape:', y_org.shape)
print('2D: ', y_work.shape)
print('after One Hot Vector:', y_all_one.shape)

#learn
x_all2 = np.insert(x_org, 0, 1.0, axis=1)

from sklearn.model_selection import train_test_split

x_train2, x_test2, y_train, y_test, \
y_train_one, y_test_one = train_test_split(
    x_all2, y_org, y_all_one, train_size=75,
    test_size=75, random_state=123)
print(x_train2.shape, x_test2.shape,
      y_train.shape, y_test.shape,
      y_train_one.shape, y_test_one.shape)

print('input data (x)')
print(x_train2[:5,:])


x, yt, x_test = x_train2, y_train_one, x_test2

#softmax function
def softmax(x):
    x = x.T
    x_max = x.max(axis=0)
    x = x - x_max
    w = np.exp(x)
    return (w / w.sum(axis=0)).T

# predicting (9.7.1, 9.7.2)
def pred(x, W):
    return softmax(x @ W)

# cross entropy (9.5.1)
def cross_entropy(yt, yp):
    return -np.mean(np.sum(yt * np.log(yp), axis=1))

from sklearn.metrics import accuracy_score
def evaluate(x_test, y_test, y_test_one, W):
    yp_test_one = pred(x_test, W)
    yp_test = np.argmax(yp_test_one, axis=1)
    loss = cross_entropy(y_test_one, yp_test_one)
    score = accuracy_score(y_test, yp_test)
    return loss, score

# initialization
M = x.shape[0]
D = x.shape[1]
N = yt.shape[1]

iters = 10000
alpha = 0.01

W = np.ones((D, N))

history = np.zeros((0,3))

for k in range(iters):
    #(9.7.1)(9.7.2)
    yp = pred(x, W)

    # (9.7.4)
    yd = yp-yt

    #(9.7.5)
    W = W - alpha * (x.T @ yd) / M

    if (k % 10 == 0):
        loss, score = evaluate(x_test, y_test, y_test_one, W)
        history = np.vstack((history,
            np.array([k, loss, score])))
        print("epoch = %d loss = %f score = %f"
              % (k, loss, score))

print(history.shape)

print('initial state: loss: %f accuracy: %f'
      % (history[0,1], history[0,2]))
print('final state: loss: %f accuracy: %f'
      % (history[-1,1], history[-1,2]))

# loss figure
plt.plot(history[:,0], history[:,1])
plt.ylim(0,1.2)
plt.grid()
plt.xlabel('iter', fontsize=14)
plt.ylabel('loss', fontsize=14)
plt.title('iter vs loss', fontsize=14)
plt.show()

# accuracy figure
plt.plot(history[:,0], history[:,2])
plt.ylim(0,1)
plt.grid()
plt.xlabel('iter', fontsize=14)
plt.ylabel('accuracy', fontsize=14)
plt.title('iter vs accuracy', fontsize=14)
plt.show()



