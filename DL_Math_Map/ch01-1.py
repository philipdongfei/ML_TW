import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from IPython.display import set_matplotlib_formats
set_matplotlib_formats('png', 'pdf')

sampleData1 = np.array([[166,58.7],[176.0,75.7],[171.0,62.1],[173.0,70.4],[169.0,60.1]])
print(sampleData1)

#figure01
for p in sampleData1:
    plt.scatter(p[0],p[1],c='k',s=50)
plt.grid()
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.show()

#figure02
plt.figure(figsize=(10,5))
for p in sampleData1:
    plt.scatter(p[0],p[1],c='k',s=50)
plt.grid()
plt.plot([0,0],[-10,80],c='k',lw=1)
plt.plot([171,171],[-10,80],c='k')
plt.plot([-10,180],[65.4,65.4],c='k')
plt.xlim(-10,180)
plt.ylim(-10,80)
plt.show()

#means
means = sampleData1.mean(axis=0)
print(means)

sampleData2 = sampleData1 - means
print(sampleData2)

#figure03
for p in sampleData2:
    plt.scatter(p[0],p[1],c='k',s=50)
plt.grid()
plt.xlabel('$X$')
plt.ylabel('$Y$')
plt.plot([-6,6],[0,0],c='k')
plt.plot([0,0],[-7.5,11],c='k')
plt.xlim(-5.2,5.2)
plt.show()

#
def L(W0, W1):
    return (5*W0**2 + 58*W1**2 - 211.2*W1 + 214.96)

#L(0,W1)
#figure04
plt.figure(figsize=(6,6))
W1 = np.linspace(0, 4, 501)
#plt.ylim(1,3)
plt.plot(W1, L(0,W1))
plt.scatter(1.82,22.69,s=30)
plt.xlabel('$W_1$')
plt.ylabel('$L(0,W_1)$')
plt.grid()
plt.xlim(0,3.5)
plt.ylim(0,200)
plt.show()

def pred1(X):
    return 1.82*X

#figure05
for p in sampleData2:
    plt.scatter(p[0],p[1],c='k',s=50)
X=np.array([-6,6])
plt.plot(X, pred1(X), lw=1)
plt.plot([-6,6],[0,0],c='k')
plt.plot([0,0],[-11,11],c='k')
plt.xlim(-5.2,5.2)
plt.grid()
plt.xlabel('$X$')
plt.ylabel('$Y$')
plt.show()

def pred2(x):
    return 1.82*x - 245.9

#figure06
for p in sampleData1:
    plt.scatter(p[0],p[1],c='k',s=50)
x=np.array([166,176])
plt.plot(x, pred2(x), lw=1)
plt.grid()
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.show()



