

import numpy as np
import matplotlib.pyplot as plt


real = np.load('realellipsoid.npy')
real = real / 2**16
real = real.astype(np.float32)
# real = np.round(real)

rmin = 29
rmax = 2 * rmin

X = real.shape[0]    # pixels
Y = real.shape[1]    # pixels
Z = real.shape[2]    # pixels

x_ = np.linspace(0,X-1,X)
y_ = np.linspace(0,Y-1,Y)
z_ = np.linspace(0,Z-1,Z)

x, y, z = np.meshgrid(x_, y_, z_, indexing='ij')



# error_matrix = np.zeros((4,4,4,4))

r=40
i=-8
j=-5
k=-14
# for r in range(60,64):
#     for i in range(-9,-5):
#         for j in range(-5,-1):
#             for k in range(-14,-10):
xc = X/2+i
yc = Y/2+j
zc = Z/2+k


ellipsoid = (x - xc)**2/1**2 + (y - yc)**2/1**2 + (z-zc)**2/2**2 >= r**2
ellipsoid = ellipsoid.astype(np.float32)*np.max(real)

# print(real.shape)
# print(ellipsoid.shape)

error = np.sum(np.abs(real - ellipsoid))
# print(f"Error: {error}")

print(f"r: {r}, i: {i}, j: {j}, k: {k}, error: {error}")
# error_matrix[i+10,j+10,k+10] = error
# error_matrix[r-60,i+9,j+5,k+14] = error
# error_matrix[r-20,i+30,j+30,k+30] = error


# print(np.min(error_matrix))
# print(np.where(error_matrix == np.min(error_matrix)))

# ind = np.where(error_matrix == np.min(error_matrix))

# rc = ind[0][0]+20
# xc = X/2-ind[1][0]-30
# yc = Y/2-ind[2][0]-30
# zc = Z/2+ind[3][0]+30


# xc = X/2
# yc = Y/2
# zc = Z/2


ellipsoid = (x - xc)**2/1**2 + (y - yc)**2/1**2 + (z-zc)**2/2**2 >= r**2
# ellipsoid = ellipsoid.astype(np.float32)/np.max(real)

for i in range(25):
    plt.subplot(5,5,i+1)
    plt.imshow(real[5*i])
    plt.imshow(ellipsoid[5*i], alpha=0.5)
# plt.subplot(3,3,1)
# plt.imshow(real[50])


np.save('estimated_ellipsoid.npy',ellipsoid*2**16)

plt.show()
