

import numpy as np
from pyellipsoid import drawing
import matplotlib.pyplot as plt


X = 1000    # microns
Y = 1000    # microns
Z = 1000    # microns

x_ = np.linspace(0,X,X + 1)
y_ = np.linspace(0,Y,Y + 1)
z_ = np.linspace(0,Z,Z + 1)

x, y, z = np.meshgrid(x_, y_, z_, indexing='ij')

r = X/4
sphr = (x - X/2)**2 + (y - Y/2)**2 + (z - Z/2)**2 <= r**2
sphr = sphr.astype(np.float32)

rd = X/4
ellipsoid = (x - X/2)**2/1**2 + (y - Y/2)**2/1**2 + (z - Z/2)**2/2**2 <= rd**2
ellipsoid = ellipsoid.astype(np.float32)

plt.imshow(sphr[50])
plt.show()

plt.imshow(ellipsoid[50])

plt.show()
# vol = np.zeros((X, Y, Y), dtype=np.float32)

np.save('sphr.npy', sphr)
np.save('ellipsoid.npy', ellipsoid)

# print(vol.shape)




