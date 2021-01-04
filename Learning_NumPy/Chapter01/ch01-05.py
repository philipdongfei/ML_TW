import numpy as np

a = [1,2,3,4,5]
print(a[1:-1])

a = np.arange(10)
print(a)
print(a[1:5])
print(a[2:8:2])
print(a[::-1])
print(a[:3])
print(a[4:])
print(a[:3],a[3:])
print(a[::2])
print(a[:])

b = np.arange(20).reshape(4,5)

print(b[1:3, 2:4])
print(b[:2,1:])
print(b[::2,:])
print(b[:,::2])
print(b[:,::-1])
print(b[::-1,::-1])




