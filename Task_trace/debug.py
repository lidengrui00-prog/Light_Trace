import numpy as np
import torch
import matplotlib.pyplot as plt

c = []
a= [[1,2,3],[4,5,6]]
b = [[3,4,5],[5,6,7]]
c.append(a)
c.append(b)
c = np.array(c)
c = c.reshape(-1,3)
print(c)
