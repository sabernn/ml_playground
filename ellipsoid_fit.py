

import numpy as np
import matplotlib.pyplot as plt


real = np.load('realellipsoid.npy')

print(real.shape)

for i in range(16):
    plt.subplot(4,4,i+1)
    plt.imshow(real[8*i])
# plt.subplot(3,3,1)
# plt.imshow(real[50])

plt.show()
