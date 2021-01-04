import numpy as np

print("argmax():")
a = np.random.randint(10, size=10)
print(a)
print(np.argmax(a))
print(a.argmax())

b = np.random.randint(10, size=(3,4))
print(b)
print(np.argmax(b))
print(np.argmax(b, axis=0))
print(np.argmax(b, axis=1))

c = np.random.randint(10, size=(2,3,4))
print(c)
print(np.argmax(c, axis=0))
print(np.argmax(c, axis=1))
print(np.argmax(c, axis=2))

print("argmin():")
d = np.array([
    [1.2,1.5,2.3,1.8],
    [0.2,2.5,2.1,2.1],
    [3.1,3.3,1.5,2.1]])
print(d)
print(np.argmin(d))
print(np.argmin(d,axis=0))
print(np.argmin(d,axis=1))
print(d.argmin(axis=1))






