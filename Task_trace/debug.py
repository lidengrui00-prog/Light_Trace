import numpy as np
import torch
import matplotlib.pyplot as plt

plt.figure(figsize=(5,5))
x = np.linspace(0,1,10)
y = 2 * x + 1
z = 5 * x + 3

plt.plot(x,y)
plt.plot(x,z)
plt.show()
