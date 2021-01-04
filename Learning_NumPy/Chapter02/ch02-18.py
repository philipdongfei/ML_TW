import numpy as np

a = np.random.randint(0,10,size=20)
print(a)
print(np.nonzero(a))
print(a.nonzero())
print(a[a.nonzero()])

b = np.random.randint(0,10,size=(4,5))
print(b)
print(np.nonzero(b))
print(b.nonzero())
print(b[b.nonzero()])


