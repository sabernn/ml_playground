

import numpy as np
import matplotlib.pyplot as plt


def generate_crack():
    img_size = 1024
    count = np.random.randint(10, 30)
    print(count)
    img = np.zeros((img_size, img_size))
    # count = 10*np.random.uniform(0, 1)
    x = np.linspace(0, 1, img_size)
    y = np.linspace(0, 1, img_size)
    yp = np.random.uniform(0.25*img_size,0.75*img_size)
    output = [count]
    for ind in range(len(img[0])):
        count = (1-abs(np.random.normal(0, 0.005)))*count
        y[ind] = yp
        yp = np.random.normal(0, 1) + yp
        # count - np.round(np.random.uniform(0, 1)*count,0)
        print(count)
        output.append(count)
        if count == 0:
            break
        # ind += 1
        indy = int(y[ind])
        img[indy-int(count/2):indy+int(count/2),ind] = 1

    return output,x,y,img


def generate_crack_3d():
    h = 1024
    w = 1024
    d = 2048
    count = np.random.randint(10, 30)
    # print(count)
    vol = np.zeros((d, h, w))
    # count = 10*np.random.uniform(0, 1)
    x = np.linspace(0, 1, w)
    y = np.linspace(0, 1, h)
    z = np.linspace(0, 1, d)
    zp = np.random.uniform(0.25*d,0.75*d)
    output = [count]
    for ind in range(w):
        count = (1-abs(np.random.normal(0, 0.005)))*count
        z[ind] = zp
        zp = np.random.normal(0, 1) + zp
        # count - np.round(np.random.uniform(0, 1)*count,0)
        # print(count)
        output.append(count)
        if count == 0:
            break
        # ind += 1
        indz = int(z[ind])
        vol[indz-int(count/2):indz+int(count/2),:,ind] = 255

    return output,x,y,z,vol


if __name__ == '__main__':
    # crack,x,y,img = generate_crack()
    crack,x,y,z,vol = generate_crack_3d()
    # plt.plot(crack)
    # plt.show()
    # plt.plot(x,y)
    # plt.show()
    # plt.imshow(img*255)
    
    plt.subplot(3,3,1)
    plt.imshow(vol[:,100,:])
    plt.subplot(3,3,2)
    plt.imshow(vol[:,200,:])
    plt.subplot(3,3,3)
    plt.imshow(vol[:,300,:])
    plt.subplot(3,3,4)
    plt.imshow(vol[:,400,:])
    plt.subplot(3,3,5)
    plt.imshow(vol[:,500,:])
    plt.subplot(3,3,6)
    plt.imshow(vol[800])
    plt.subplot(3,3,7)
    plt.imshow(vol[900])
    plt.subplot(3,3,8)
    plt.imshow(vol[1000])
    plt.subplot(3,3,9)
    plt.imshow(vol[1020])



    plt.show()


    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # X,Y,Z = vol
    # ax.scatter(X, Y, Z, c='r', marker='o')
    # plt.show()

