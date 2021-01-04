import numpy as np

a = np.arange(12)
b = np.arange(2)

print(a)
print(b)
print(np.hstack((a,b)))

c = np.arange(4).reshape(2,2)
print(c)
#print(np.hstack((a,c)))
d = np.arange(6).reshape(2,3)
print(d)
print(np.hstack((c,d)))
print(np.hstack((c,d)).shape)

e = np.arange(12).reshape(2,2,3)
f = np.arange(6).reshape(2,1,3)
print(e)
print(f)
print(np.hstack((e,f)))

a = np.arange(12).reshape(-1,1)
b = np.arange(2).reshape(-1,1)
print(a)
print(b)
print(np.vstack((a,b)))

c = np.arange(2).reshape(1,2)
print(c)
#print(np.vstack((a,c)))
d = np.arange(4).reshape(2,2)
print(d)
print(np.vstack((c,d)))


e = np.arange(24).reshape(4,3,2)
f = np.arange(6).reshape(1,3,2)
print(e)
print(f)
g = np.vstack((e,f))
print(g)
print(g.shape)






