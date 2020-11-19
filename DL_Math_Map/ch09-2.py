#use axis in maxtrix
import numpy as np

x = np.array([[1,2,3],[4,5,6]])
print(x)

y = x.sum(axis=0)
print(y)

z = x.sum(axis=1)
print(z)

w = x.sum()
print(w)

