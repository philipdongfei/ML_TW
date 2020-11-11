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

u = np.linspace(-5, 5, 501)
v = np.linspace(-5, 5, 501)

U, V = np.meshgrid(u, v)
Z = L(U, V)

W = np.array([4.0, 4.0])
W1 = [W[0]]
W2 = [W[1]]
N = 21
alpha = 0.05
for i in range(N):
    W = W - alpha * np.array([Lu(W[0], W[1]), Lv(W[0], W[1])])
    W1.append(W[0])
    W2.append(W[1])

#fig-04-07-1
n_loop = 0

WW1 = np.array(W1[:n_loop])
WW2 = np.array(W2[:n_loop])
ZZ = L(WW1, WW2)
fig = plt.figure(figsize=(8,8))
ax = plt.axes(projection='3d')
ax.set_zlim(0,250)
ax.view_init(50, 240)
ax.set_xlabel('u', fontsize=14)
ax.set_ylabel('v', fontsize=14)
ax.xaxis._axinfo["grid"]['linewidth'] = 2.
ax.yaxis._axinfo["grid"]['linewidth'] = 2.
ax.zaxis._axinfo["grid"]['linewidth'] = 2.
ax.contour3D(U, V, Z, 100, cmap='Blues', alpha=1.0)
#ax.plot_surface(U, V, Z, cmap='Blues', linewidth=0)
plt.show()

#fig04-07-2
plt.figure(figsize=(8,8))
plt.contourf(U, V, Z, levels=[5,10,20,30,40,50,70,100,200], cmap='Blues')
C = plt.contour(U, V, Z, levels=[5,10,20,30,40,50,70,100,200],colors='k')
plt.clabel(C, inline=1, fontsize=10, fmt='%r')
plt.gca().set_aspect('equal')
plt.xticks(range(-4,5,1))
plt.yticks(range(-4,5,1))
plt.xlabel('u',fontsize=14)
plt.ylabel('v',fontsize=14)
plt.grid(linewidth=2)
plt.show()

n_loop = 2
#fig04-08-1
WW1 = np.array(W1[:n_loop])
WW2 = np.array(W2[:n_loop])
ZZ = L(WW1, WW2)
fig = plt.figure(figsize=(8,8))
ax = plt.axes(projection='3d')
ax.set_zlim(0, 250)
ax.set_xlabel('u', fontsize=14)
ax.set_ylabel('v', fontsize=14)
ax.set_zlabel('L(u,v)', fontsize=14)
ax.view_init(50, 240)
ax.xaxis._axinfo["grid"]['linewidth'] = 2.
ax.yaxis._axinfo["grid"]['linewidth'] = 2.
ax.zaxis._axinfo["grid"]['linewidth'] = 2.
ax.contour3D(U, V, Z, 100, cmap='Blues', alpha=0.7)
ax.plot3D(WW1, WW2, ZZ, 'o-', c='k', alpha=1, markersize=7)
plt.show()

# fig04-08-2
plt.figure(figsize=(8,8))
plt.contourf(U, V, Z, levels=[5,10,20,30,40,50,70,100,200], cmap='Blues')
C = plt.contour(U, V, Z, levels=[5,10,20,30,40,50,70,100,200],colors='k')
plt.clabel(C, inline=1, fontsize=10, fmt='%r')
plt.plot(W1[:n_loop], W2[:n_loop], '-o', c='k')
plt.gca().set_aspect('equal')
plt.xticks(range(-4,5,1))
plt.yticks(range(-4,5,1))
plt.xlabel('u', fontsize=14)
plt.ylabel('v', fontsize=14)
plt.grid(linewidth=2)
plt.show()


for n_loop in range(3, 22):
    # fig04-09-n
    WW1 = np.array(W1[:n_loop])
    WW2 = np.array(W2[:n_loop])
    ZZ = L(WW1, WW2)
    fig = plt.figure(figsize=(8,8))
    ax = plt.axes(projection='3d')
    ax.set_zlim(0, 250)
    ax.set_xlabel('u', fontsize=14)
    ax.set_ylabel('v', fontsize=14)
    ax.set_zlabel('L(u,v)', fontsize=14)
    ax.view_init(50, 240)
    ax.xaxis._axinfo["grid"]['linewidth'] = 2.
    ax.yaxis._axinfo["grid"]['linewidth'] = 2.
    ax.zaxis._axinfo["grid"]['linewidth'] = 2.
    ax.contour3D(U, V, Z, 100, cmap='Blues', alpha=0.7)
    ax.plot3D(WW1, WW2, ZZ, 'o-', c='k', alpha=1, markersize=7)
    plt.show()

    #fig04-09-n
    plt.figure(figsize=(8,8))
    plt.contourf(U, V, Z, levels=[5,10,20,30,40,50,70,100,200], cmap='Blues')
    C = plt.contour(U, V, Z, levels=[5,10,20,30,40,50,70,100,200],colors='k')
    plt.clabel(C, inline=1, fontsize=10, fmt='%r')
    plt.plot(W1[:n_loop], W2[:n_loop], '-o', c='k')
    plt.gca().set_aspect('equal')
    plt.xticks(range(-4,5,1))
    plt.yticks(range(-4,5,1))
    plt.xlabel('u', fontsize=14)
    plt.ylabel('v', fontsize=14)
    plt.grid(linewidth=2)
    plt.show()









