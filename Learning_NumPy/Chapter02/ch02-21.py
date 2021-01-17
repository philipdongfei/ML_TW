import numpy as np

a = np.array([1,2,3])
b = np.array([4,5,6])

print(np.r_[a,b])
print(np.r_[2,5,3,np.array([2,3]), 4.2])
c = np.zeros((2,3))
d = np.ones((3,3))

print(np.r_[c,d])
d = np.ones((3,4))
"print(np.r_[c,d])"

a = np.ones((2,2))
b = np.zeros((2,2))

print(np.r_['1',a,b])
print(np.r_['0',a,b])
print(np.r_['0',a,b].shape)
print(np.r_[a,b].shape)

c = np.ones((2,2,2))
d = np.zeros((2,2,2))
print(c)
print(d)
print(np.r_['0',c,d])
print(np.r_['1',c,d])
print(np.r_['2',c,d])
print(np.r_['0',c,d].shape)
print(np.r_['1',c,d].shape)
print(np.r_['2',c,d].shape)

print(np.r_['0,2',[0,1,2],[3,3,3]])
print(np.r_['0,2',[0,1,2],[3,3,3]].shape)
print(np.r_['0,3',[0,1,2],[3,3,3]])
print(np.r_['0,3',[0,1,2],[3,3,3]].shape)

print(np.r_['0,2,-1',[0,1,2],[3,3,3]])
print(np.r_['0,2,-1',[0,1,2],[3,3,3]].shape)
print(np.r_['0,2,0',[0,1,2],[3,3,3]])
print(np.r_['0,2,0',[0,1,2],[3,3,3]].shape)
print(np.r_['0,3,0',[0,1,2],[3,3,3]])
print(np.r_['0,3,0',[0,1,2],[3,3,3]].shape)
print(np.r_[0:10])
print(np.r_[:10])
print(np.r_[0:10:2])
print(np.r_[10:0:-1])
print(np.r_[0:10,0,4,np.array([3,3])])
a = np.array([1,4,6])
b = np.array([2,2,2])
print(np.r_['r',a,b])
print(np.r_['c',a,b])
c = np.ones((4,5))
d = np.zeros((2,5))
print(np.r_['r',c,d])
print(np.r_['c',c,d])
a = np.ones((3,2))
b = np.ones((3,3))
print(a)
print(b)
print(np.c_[a,b])
c = np.zeros(3)
print(c)
print(np.c_[a,c])
print(np.c_[a,c].shape)