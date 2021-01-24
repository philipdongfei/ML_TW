import numpy as np
import numpy.linalg as LA

a = np.array([[2,3],[4,-1]])
print(a)
print(LA.det(a))

d = np.random.randint(-5,6,size=(3,3,3))
print(d)
print(LA.det(d))


