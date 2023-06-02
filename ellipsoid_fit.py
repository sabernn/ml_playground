

import numpy as np
import matplotlib.pyplot as plt


real = np.load('realellipsoid.npy')
real = real / (2**16)

rmin = 50
rmax = 2 * rmin

X = real.shape[0]    # pixels
Y = real.shape[1]    # pixels
Z = real.shape[2]    # pixels

x_ = np.linspace(0,X-1,X)
y_ = np.linspace(0,Y-1,Y)
z_ = np.linspace(0,Z-1,Z)

x, y, z = np.meshgrid(x_, y_, z_, indexing='ij')

xc = X/2
yc = Y/2
zc = Z/2

ellipsoid = (x - xc)**2/1**2 + (y - yc)**2/1**2 + (z-zc)**2/2**2<= rmin**2
ellipsoid = ellipsoid.astype(np.int8)

print(real.shape)

for i in range(25):
    plt.subplot(5,5,i+1)
    plt.imshow(real[5*i], cmap='gray')
    plt.imshow(ellipsoid[5*i], alpha=0.5)
# plt.subplot(3,3,1)
# plt.imshow(real[50])

plt.show()
