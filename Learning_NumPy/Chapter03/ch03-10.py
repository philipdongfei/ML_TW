import numpy as np

a = np.array([1,2])
b = np.array([4,3])
print(np.dot(a,b))
print(np.dot(4,5))

e = np.matrix([1,2])
f = np.matrix([[4],[3]])
print(np.dot(e,f))

a = np.array([[1,2],[3,4]])
b = np.array([[4,3],[2,1]])
print(np.dot(a,b))
print(np.dot(b,a))

c = np.arange(9).reshape((3,3))
d = np.ones((3,3))
print(np.dot(c,d))

e = np.matrix([[0,1,2],[3,4,5],[6,7,8]])
f = np.matrix([[1,1,1],[1,1,1],[1,1,1]])
print(np.dot(e,f))



