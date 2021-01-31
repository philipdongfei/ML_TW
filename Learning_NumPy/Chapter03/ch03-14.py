import numpy as np

a = np.random.randint(-9,10,size=(2,2))
print(a)
print(np.linalg.inv(a))

print(np.dot(a, np.linalg.inv(a)))

b = np.random.randint(-10,10,size=(3,3))
print(b)
c = np.linalg.inv(b)
print(np.dot(b,c))

