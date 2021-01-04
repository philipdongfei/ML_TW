import numpy as np

print(np.ones(3))
print(np.ones((2,3)))

print(np.ones(4,dtype=np.int8))

a = np.array([[1,2,3],[2,3,4]])
print(np.ones_like(a))

b = np.array([2,3,4],dtype="int8")
print(np.ones_like(b))

