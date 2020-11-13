import numpy as np
import matplotlib.pyplot as plt

def std(x, sigma=1):
    return (np.exp(-(x/sigma)**2/2))/(np.sqrt(2*np.pi)*sigma)

def sigmoid(x):
    return (1/(1+np.exp(x)))

x = np.linspace(-5, 5, 1000)
y_std = std(x, 1.6)
sig = sigmoid(x)
y_sig = sig * (1-sig)

plt.figure(figsize=(8,8))
plt.plot(x, y_std, label='std', c='k', lw=3, linestyle='-.')
plt.plot(x, y_sig, label="sig", c='b', lw=3)
plt.legend(fontsize=14)
plt.grid(lw=2)
plt.show()

