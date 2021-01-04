import numpy as np

a = np.arange(10).reshape(2,5)
print(a)
b = a.flatten()
print(a)
print(b)
print(a.shape)
print(b.shape)

c = np.arange(12).reshape(2,2,3)
print(c)
d = c.flatten()
print(d)
print(c.shape)
print(d.shape)

