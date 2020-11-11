import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

#pdf
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('png', 'pdf')

def L(u, v):
    return 3 * u**2 + 3 * v**2 - u*v + 7*u - 7*v + 10

def Lu(u, v):
    return 6 * u - v + 7

def Lv(u, v):
    return -u + 6 * v - 7

u = np.linspace(0.2, 5, 21)
v = np.linspace(0.2, 5, 21)
U, V = np.meshgrid(u, v)
Z = L(U, V)
uu = np.linspace(-5, 5, 41)
vv = np.linspace(-5, 5, 41)
zz = np.zeros(uu.shape)
Luu = L(uu, zz)
Lvv = L(zz, vv)
uu2 = np.vstack((uu, uu))
vv2 = np.vstack((vv, vv))
zz2 = np.vstack((zz, zz))
Luu2 = np.vstack((Luu, zz))
Lvv2 = np.vstack((Lvv, zz))

fig = plt.figure(figsize=(8,8))
ax = plt.axes(projection='3d')
ax.set_zlim(0,250)
ax.view_init(50,240)
ax.set_xlabel('$u$', fontsize=14)
ax.set_ylabel('$v$', fontsize=14)
ax.set_zlabel('$L(u,v)$', fontsize=14)
ax.xaxis._axinfo["grid"]['linewidth'] = 2.
ax.yaxis._axinfo["grid"]['linewidth'] = 2.
ax.zaxis._axinfo["grid"]['linewidth'] = 2.
ax.set_xlim(-5,5)
ax.set_ylim(-5,5)
ax.plot_surface(U, V, Z, rstride=1, cstride=1, cmap='Blues',
                linewidth=0, shade=False, antialiased=False)
ax.plot_surface(uu2, zz2, Luu2, color='white', linewidth=0, shade=False)
ax.plot_surface(zz2, vv2, Lvv2, color='white', linewidth=0, shade=False)
ax.plot3D(uu, zz, Luu, c='k', lw=3, linestyle='-', label='$z = L(u, 0)$')
ax.plot3D(zz, vv, Lvv, c='k', lw=3, linestyle='--', label='$z = L(0, v)$')
ax.legend()
plt.show()






