import numpy as np
import matplotlib.pyplot as plt


div = 8
dim = 8

p = [-1, 1, -3, 8, -7]

xMin = -2
xMax = 1

x = np.linspace(xMin, xMax, num=div)

xx = np.linspace(xMin, xMax, num=div*10)

y = np.polyval(p, x)
yy = np.polyval(p, xx)

z = y + 5 * np.random.randn(div)

def print_fix(x):
    [print('{:.3f}'.format(n)) for n in x]

def print_fix_model(m):
    w = m.coef_.tolist()
    w[0] = m.intercept_
    print_fix(w)

def f(x):
    return [x**i for i in range(dim)]

X = [f(x0) for x0 in x]

XX = [f(x0) for x0 in xx]

from sklearn.linear_model import LinearRegression

model = LinearRegression().fit(X, z)

yy_pred = model.predict(XX)

from sklearn.linear_model import Ridge

model2 = Ridge(alpha=0.5).fit(X, z)

yy_pred2 = model2.predict(XX)

plt.figure(figsize=(8,6))
plt.plot(xx, yy, label='polynomial', lw=1, c='k')
plt.scatter(x, z, label='observed', s=50, c='k')
plt.plot(xx, yy_pred, label='linear regression', lw=3, c='k')
plt.plot(xx, yy_pred2, label='L2 regularizer', lw=3, c='b')
plt.xlabel('X')
plt.ylabel('Y')
plt.xticks(size=14)
plt.yticks(size=14)
plt.grid(lw=2)
plt.legend(fontsize=14)
plt.show()
