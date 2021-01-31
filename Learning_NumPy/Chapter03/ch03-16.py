import numpy as np

a = np.array([1,2,3])
b = np.array([5,4,0])
print(np.cross(a,b))

c = np.array([-1,1,3])
d = np.array([2,3,3])
print(np.cross(c,d))

b_2 = np.array([5,4])
print(np.cross(a,b_2))

ac = np.vstack((a,c))
bd = np.vstack((b,d))
print(ac)
print(bd)
print(np.cross(ac,bd))

ac_2 = ac.transpose()
print(ac_2)
print(np.cross(ac_2, bd,axisa=0))
bd_2 = bd.transpose()
print(bd_2)
print(np.cross(ac,bd_2,axisb=0))
print(np.cross(ac,bd,axisc=1))
#print(np.cross(ac,bd,axisc=0))
print(np.cross(ac_2, bd_2,axis=0))


