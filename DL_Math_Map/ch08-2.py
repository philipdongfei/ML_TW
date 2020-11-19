import numpy as np
import matplotlib.pyplot as plt

from IPython.display import set_matplotlib_formats
set_matplotlib_formats('png', 'pdf')

from sklearn.datasets import load_iris
iris = load_iris()
x_org, y_org = iris.data, iris.target
print('shape:', x_org.shape, y_org.shape)

# sepal_length and sepal_width
x_data, y_data = iris.data[:100, :2], iris.target[:100]
print('shape:', x_data.shape, y_data.shape)

x_data = np.insert(x_data, 0, 1.0, axis=1)
print('after add the data: ', x_data.shape)

print(x_data.shape, y_data.shape)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x_data, y_data, train_size=70, test_size=30,
    random_state=123)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

# diagram of distribution
x_t0 = x_train[y_train == 0]
x_t1 = x_train[y_train == 1]
plt.figure(figsize=(6,6))
plt.scatter(x_t0[:,1], x_t0[:,2], marker='x', c='b', label='0 (setosa)')
plt.scatter(x_t1[:,1], x_t1[:,2], marker='o', c='k', label='1 (versicolor)')
plt.xlabel('sepal_length', fontsize=14)
plt.ylabel('sepal_width', fontsize=14)
plt.xticks(size=16)
plt.yticks(size=16)
plt.legend(fontsize=16)
plt.show()


# diagram of distribution
x_t0 = x_train[y_train == 0]
x_t1 = x_train[y_train == 1]
plt.figure(figsize=(6,6))
plt.scatter(x_t0[:,1], x_t0[:,2], marker='x', s=50, c='b', label='yt = 0')
plt.scatter(x_t1[:,1], x_t1[:,2], marker='o', s=50, c='k', label='yt = 1')
plt.xlabel(r'$x_1$', fontsize=16)
plt.ylabel(r'$x_2$', fontsize=16)
plt.xticks(size=16)
plt.yticks(size=16)
plt.legend(fontsize=16)
plt.show()

x = x_train
yt = y_train

print(x[:5])

def sigmoid(x):
    return 1/(1+np.exp(-x))

def pred(x, w):
    return sigmoid(x @ w)

def cross_entropy(yt, yp):
    cel = -(yt * np.log(yp) + (1 - yt) * np.log(1 - yp))
    return (np.mean(cel))

def classify(y):
    return np.where(y < 0.5, 0, 1)

from sklearn.metrics import accuracy_score
def evaluate(xt, yt, w):
    yp = pred(xt, w)
    loss = cross_entropy(yt, yp)
    yp_b = classify(yp)
    score = accuracy_score(yt, yp_b)
    return loss, score

#initial process
M = x.shape[0]
D = x.shape[1]

iters = 10000
alpha = 0.01
w = np.ones(D)
history = np.zeros((0,3))

for k in range(iters):
    yp = pred(x, w)
    yd = yp - yt
    w = w - alpha * (x.T @ yd) / M

    if (k % 10 == 0):
        loss, score = evaluate(x_test, y_test, w)
        history = np.vstack((history,
                             np.array([k, loss, score])))
        print("iter = %d loss = %f score = %f"
              % (k, loss, score))

print('initial state: loss function: %f precise: %f'
      % (history[0,1], history[0,2]))
print('final state: loss function: %f precise: %f'
      % (history[-1,1], history[-1,2]))

x_t0 = x_test[y_test==0]
x_t1 = x_test[y_test==1]

def b(x, w):
    return (-(w[0] + w[1] * x)/ w[2])

x1 = np.asarray([x[:,1].min(), x[:,1].max()])
y1 = b(x1, w)


plt.figure(figsize=(6,6))
plt.scatter(x_t0[:,1], x_t0[:,2], marker='x',
            c='b', s=50, label='class 0')
plt.scatter(x_t1[:,1], x_t1[:,2], marker='o',
            c='k', s=50, label='class 1')
plt.plot(x1, y1, c='b')
plt.xlabel('sepal_length', fontsize=14)
plt.ylabel('sepal_width', fontsize=14)
plt.xticks(size=16)
plt.yticks(size=16)
plt.legend(fontsize=16)
plt.show()

# figure loss
plt.figure(figsize=(6,4))
plt.plot(history[:,0], history[:,1], 'b')
plt.xlabel('iter', fontsize=14)
plt.ylabel('cost', fontsize=14)
plt.title('tier vs cost', fontsize=14)
plt.show()

# figure precise
plt.figure(figsize=(6,4))
plt.plot(history[:,0], history[:,2], 'b')
plt.xlabel('iter', fontsize=14)
plt.ylabel('accuracy', fontsize=14)
plt.title('iter vs accuracy', fontsize=14)
plt.show()


from mpl_toolkits.mplot3d import Axes3D
x_1 = np.linspace(4, 7.5, 100)
x_2 = np.linspace(2, 4.5, 100)
xx1, xx2 = np.meshgrid(x_1, x_2)
xxx = np.asarray([np.ones(xx1.ravel().shape),
                  xx1.ravel(), xx2.ravel()]).T
c = pred(xxx, w).reshape(xx1.shape)
plt.figure(figsize=(8,8))
ax = plt.subplot(1,1,1,projection='3d')
ax.plot_surface(xx1,xx2,c,color='blue',
    edgecolor='black', rstride=10, cstride=10, alpha=0.1)
ax.scatter(x_t1[:,1], x_t1[:,2], 1, s=20, alpha=0.9, marker='o', c='b')
ax.scatter(x_t0[:,1], x_t0[:,2], 0, s=20, alpha=0.9, marker='s', c='b')
ax.set_xlim(4,7.5)
ax.set_ylim(2,4.5)
ax.view_init(elev=20, azim=60)
plt.show()



from sklearn.linear_model import LogisticRegression
from sklearn import svm

model_lr = LogisticRegression(solver='liblinear')
model_svm = svm.SVC(kernel='linear')

model_lr.fit(x, yt)
model_svm.fit(x, yt)


lr_w0 = model_lr.intercept_[0]
#x1(sepal_length) ratio
lr_w1 = model_lr.coef_[0,1]
#x2(sepal_width) ratio
lr_w2 = model_lr.coef_[0,2]

#SVM
svm_w0 = model_svm.intercept_[0]
# x1(sepal_length) ratio
svm_w1 = model_svm.coef_[0,1]
# x2(sepal_width) ratio
svm_w2 = model_svm.coef_[0,2]

def r1(x):
    wk = lr_w0 + lr_w1 * x
    wk2 = -wk / lr_w2
    return (wk2)

def svm(x):
    wk = svm_w0 + svm_w1 * x
    wk2 = -wk / svm_w2
    return (wk2)

y_r1 = r1(x1)
y_svm = svm(x1)


print(x1, y1, y_r1, y_svm)

# distribution figure
fig = plt.figure(figsize=(6,6))
ax = fig.add_subplot(1,1,1)
plt.scatter(x_t0[:,1], x_t0[:,2],marker='x', c='b')
plt.scatter(x_t1[:,1], x_t1[:,2],marker='o', c='k')
ax.plot(x1, y1, linewidth=2, c='k', label='Hands On')
# lr model
ax.plot(x1, y_r1, linewidth=2, c='k', linestyle='--', label='scikit LR')
# svm
ax.plot(x1, y_svm, linewidth=2, c='k', linestyle='-.', label='scikit SVM')

ax.legend()
ax.set_xlabel('$x_1$', fontsize=16)
ax.set_ylabel('$x_2$', fontsize=16)
plt.show()




