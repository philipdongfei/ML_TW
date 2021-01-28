import numpy as np
import numpy.linalg as LA

a = np.array([[1,0],[0,2]])
print(a)
print(LA.eig(a))

b = np.array([[2,5],[3,-8]])
print(b)
print(LA.eig(b))

c = np.random.randint(-10,10,size=(3,3))
w,v = LA.eig(c)
print(w)
print(v)

c = np.random.randint(-10,10,size=(3,3,3))
print(c)
w,v = LA.eig(c)
print(w)
print(v)

