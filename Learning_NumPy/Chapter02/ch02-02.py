import numpy as np

print("append:")
a = np.arange(12)
print(np.append(a, [6,4,2]))

b = np.arange(12).reshape((3,4))
print(b)
print(np.append(b, [1,2,3,4]))

print(np.append(b, [[12,13,14,15]], axis=0))
#print(np.append(b, [12,13,14,15], axis=0)) #ValueError

c = np.arange(12).reshape((3,4))
print(c)

d = np.linspace(0, 26, 12).reshape(3,4)
print(d)
print(np.append(c,d, axis=0))
print(np.append(c,d, axis=1))


