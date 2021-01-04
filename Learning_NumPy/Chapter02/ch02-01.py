import numpy as np

a = np.arange(12)
print(a)

print("reshape:")
b = np.reshape(a, (3,4))
print(b)

b[0,1] = 0
print(b)
print(a)

c = np.arange(12)

d = np.reshape(c, (3,4), order='C')
print(d)

d = np.reshape(c, (3,4), order='F')
print(d)

#np.reshape(c, (3,5)) # ValueError

a = np.arange(12)
print(np.reshape(a, (3,-1)))
print(np.reshape(a, (-1,6)))

a = np.arange(12).reshape((3,4))
print(a)

b = np.arange(12).reshape((3,-1))
print(b)

#c = np.arange(15).reshape((3,4))

print("resize:")
a = np.arange(12)

print(np.reshape(a, (3,4)))
print(np.resize(a, (3,4)))
print(np.resize(a, (3,2)))

b = np.resize(a, (3,4))
print(b)
b[0,1] = 0
print(b)
print(a)



