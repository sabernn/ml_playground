

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
    countz = np.random.randint(10, 30)
    # print(count)
    vol = np.zeros((d, h, w))
    # count = 10*np.random.uniform(0, 1)
    x = np.linspace(0, 1, w)
    y = np.linspace(0, 1, h)
    z = np.linspace(0, 1, d)
    zp = np.random.uniform(0.25*d,0.75*d)
    output = [countz]
    for ind in range(w):
        countz = (1-abs(np.random.normal(0, 0.005)))*countz
        z[ind] = zp
        zp = np.random.normal(0, 1) + zp
        # count - np.round(np.random.uniform(0, 1)*count,0)
        # print(count)
        output.append(countz)
        if countz == 0:
            break
        # ind += 1
        indz = int(z[ind])
        vol[indz-int(countz/2):indz+int(countz/2),:,ind] = 255

    return output,x,y,z,vol

def crack_surface():
    img_size = 1024
    height = 10
    width = 200
    count_r = height
    count_l = height
    img = np.zeros((img_size, img_size))
    count = 0
    while count_r > 0.1 and count_l > 0.1:
        count_r *= (1-abs(np.random.normal(0, 0.005)))
        count_l *= (1-abs(np.random.normal(0, 0.005)))
        print(count_r,count_l)
        x = int(img_size/2)
        y = int(img_size/2)
        img[y-int(count_l):y+int(count_r),x-count:x+count] = 255
        count += 1

    return img

    


if __name__ == '__main__':
    # crack,x,y,img = generate_crack()
    # crack,x,y,z,vol = generate_crack_3d()

    scrack = crack_surface()
    # plt.plot(crack)
    # plt.show()
    # plt.plot(x,y)
    # plt.show()
    # plt.imshow(img*255)

    plt.figure(figsize=(10,10))
    plt.imshow(scrack)
    plt.show()
    
    plt.figure(figsize=(10,10))
    for i in range(9):
        plt.subplot(3,3,i+1)
        plt.imshow(vol[:,i*100,:])

    plt.show()


    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.voxels(vol[::100,::100,::100])
    ax.set_xlabel('Z (build direction)')
    ax.set_ylabel('Y (image height)')
    ax.set_zlabel('X (image width)')

    plt.show()

