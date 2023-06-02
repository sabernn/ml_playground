

import numpy as np
import matplotlib.pyplot as plt


real = np.load('realellipsoid.npy')
real = real / 2**16
real = real.astype(np.float32)*3
real = np.round(real)

rmin = 29
rmax = 2 * rmin

X = real.shape[0]    # pixels
Y = real.shape[1]    # pixels
Z = real.shape[2]    # pixels

x_ = np.linspace(0,X-1,X)
y_ = np.linspace(0,Y-1,Y)
z_ = np.linspace(0,Z-1,Z)

x, y, z = np.meshgrid(x_, y_, z_, indexing='ij')

error_matrix = np.zeros((4,4,4))

for i in range(-2,2):
    for j in range(-2,2):
        for k in range(-2,2):
            xc = X/2+i
            yc = Y/2+j
            zc = Z/2+k


            ellipsoid = (x - xc)**2/1**2 + (y - yc)**2/1**2 + (z-zc)**2/2**2 >= rmin**2
            ellipsoid = ellipsoid.astype(np.float32)

        # print(real.shape)
            # print(ellipsoid.shape)

            error = np.sum(np.abs(real - ellipsoid))
            # print(f"Error: {error}")

            print(f"i: {i}, j: {j}, k: {k}, error: {error}")
            error_matrix[i+2,j+2,k+2] = error


print(np.min(error_matrix))
print(np.where(error_matrix == np.min(error_matrix)))

ind = np.where(error_matrix == np.min(error_matrix))

xc = X/2-ind[0][0]-2
yc = Y/2-ind[1][0]-2
zc = Z/2+ind[2][0]-2

# xc = X/2
# yc = Y/2
# zc = Z/2


ellipsoid = (x - xc)**2/1**2 + (y - yc)**2/1**2 + (z-zc)**2/2**2 >= rmin**2
ellipsoid = ellipsoid.astype(np.float32)

for i in range(25):
    plt.subplot(5,5,i+1)
    plt.imshow(real[5*i], cmap='gray')
    plt.imshow(ellipsoid[5*i], alpha=0.5)
# plt.subplot(3,3,1)
# plt.imshow(real[50])

plt.show()
