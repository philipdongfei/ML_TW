import numpy as np

a = np.array([1,2,3,2,1])
b = np.array([0,2,4,6,8,1])
print(a.shape)
print(b.shape)

print(np.outer(a,b))

b = b.reshape(2,-1)
c = np.random.randint(0,5,size=(2,4))
print(b)
print(c)
print(np.outer(b,c))
print(np.outer(b.ravel(),c.ravel()))

