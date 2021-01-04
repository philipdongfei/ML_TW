import numpy as np

a = np.array([0,1,2])
print(a.dtype)

b = np.array([0,1,2], dtype='int32')
print(b.dtype)
print(b)

c = np.array([0,1,2], dtype='float')
print(c)

#d = np.array([3e50,4e35], dtype='int64') # overflowerror dtype
d = np.array([3e50,4e35], dtype='float64')

e = np.array([3.5, 4.2, -4.3], dtype='int')
print(e)

f = np.array([0, 3, 0, -1], dtype='bool')
print(f)


