import numpy as np
import matplotlib.pyplot as plt

xx = np.linspace(0.2, 2.0, 100)
yy1 = np.log(xx) / np.log(2.0)
yy2 = np.log(xx)
yy3 = np.log(xx) / np.log(6.0)

# fig05-09
plt.figure(figsize=(6,6))
plt.ylim(-1.0, 1.0)
plt.xlim(0.2, 2.0)
plt.xlabel('$x$', fontsize=18)
plt.ylabel('$y$', fontsize=18)
plt.grid(lw=2)
plt.plot([0.2,2.0],[-0.8,1.0], 'b-', lw=2, c='black')
plt.plot(xx,yy1, linestyle='dotted', c='black', lw=2, label='$log_{2}x$')
plt.plot(xx, yy2, linestyle='solid', c='b', lw=2, label='$log_{e}x$')
plt.plot(xx, yy3, linestyle='dashed', c='black', lw=2, label='$log_{6}x$')
plt.scatter(1.0, 0.0, s=50)
plt.xticks(size=16)
plt.yticks(size=16)
plt.legend(fontsize=18)
plt.show()

