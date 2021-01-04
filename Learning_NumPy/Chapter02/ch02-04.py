import numpy as np

print("where():")
a = np.arange(20, 0, -2)
print(a)
print(np.where(a<10))
print(a[np.where(a < 10)])

a = np.arange(12).reshape((3,4))
print(a)
print(np.where(a % 2 == 0))
print(a[np.where(a % 2 == 0)])

print(np.where(a%2==0, 'even', 'odd'))
#print(np.where(a%2==0, 'even')) #ValueError

b = np.reshape(a, (3,4))
print(b)

c = b**2
print(c)
print(np.where(b%2==0, b, c))



