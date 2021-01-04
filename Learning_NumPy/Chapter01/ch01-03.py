import numpy as np

a = np.array([[1,2,3],[4,5,6]])

print(a.shape)
print(a.ndim)

a = np.array([[0,1],[2,3],[4,5]])
print(a)
print(a.shape)

b = np.array([a,a])
print(b.shape)
print(b)

print(b.sum(axis=0))
print(b.sum(axis=1))
print(b.sum(axis=2))

