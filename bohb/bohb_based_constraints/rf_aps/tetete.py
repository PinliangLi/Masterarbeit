import numpy as np

a = np.array([1, 2, 3, 4, 5])
a.reshape((-1, 1))
print(a[-1])
a.reshape((1, -1))
print(a.shape)
a.flatten()
print(a.shape)