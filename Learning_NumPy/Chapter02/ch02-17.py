import numpy as np

a = np.arange(10).reshape(2,5)

print('a:')
print(a)

print(a.ravel())
print(np.ravel(a))
print(a.ravel(order='C'))
print(a.ravel(order='F'))
print('b:')
b = np.arange(10).reshape(2,5,order='F')
print(b)
print(b.ravel(order='F'))
print(b.ravel(order='A'))
print(b.ravel())






