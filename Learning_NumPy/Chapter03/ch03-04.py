import numpy as np

a = np.random.randint(0,10,size=(2,5))
print(a)
print(np.sum(a))
b = np.array([2,4,1,6])
print(np.sum(b))
c = np.random.randint(0,10,size=(2,4,5))
print(c)
print(np.sum(c))
print(a)
print(np.sum(a,axis=0))
print(np.sum(a,axis=1))
print(c)
print(np.sum(c,axis=0))
print(np.sum(c,axis=1))
print(np.sum(c,axis=2))
print(np.sum(c,axis=0,keepdims=True))
print(np.sum(c,axis=1,keepdims=True))
print(np.sum(c,axis=2,keepdims=True))



