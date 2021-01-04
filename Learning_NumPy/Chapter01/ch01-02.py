import numpy as np

a = np.array([1,2,3])

print(type(a))

print(a)

a.data

print(a.dtype)
print(a.ndim)
print(a.size)
print(a.itemsize)
print(a.nbytes)
print(a.strides)


b = np.array([[1,2,3],[4,5,6]])
print(b)
print(b.T)

a = np.array([[0,1],[2,3],[4,5]])
print('a shape', a.shape)
print('a ndim', a.ndim)

b = np.array([a,a])
print('b shape', b.shape)
print('b ndim', b.ndim)



