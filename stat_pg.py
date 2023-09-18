

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

def crack_surface(img_size: int, center: tuple):
    # img_size = 1024
    height = 5
    width = 200
    count_ul = height
    count_ur = height
    count_ll = height
    count_lr = height
    img = np.zeros((img_size, img_size))
    count = 0
    cx = center[0]
    cyl = center[1]
    cyr = center[1]
    smoothness_factor = 1 # higher = smoother
    while count_ul > 1 and count_ur > 1 and count_ll > 1 and count_lr > 1:
        # random asymmetrical decay of crack width
        if count%smoothness_factor == 0:
            count_ul *= (1-abs(np.random.normal(0, 0.01*smoothness_factor)))
            count_ur *= (1-abs(np.random.normal(0, 0.01*smoothness_factor)))
            count_ll *= (1-abs(np.random.normal(0, 0.01*smoothness_factor)))
            count_lr *= (1-abs(np.random.normal(0, 0.01*smoothness_factor)))
            cyl += np.random.normal(0, 1)
            cyr += np.random.normal(0, 1)
        img[int(cyl-count_ll):int(cyl+count_ul),cx-count:cx-count+1] = 255
        img[int(cyr-count_lr):int(cyr+count_ur),cx+count:cx+count+1] = 255
        count += 1

    return img



if __name__ == '__main__':
    # crack,x,y,img = generate_crack()
    # crack,x,y,z,vol = generate_crack_3d()

    scrack = crack_surface(img_size=1024,
                            center=(512,512))
    # plt.plot(crack)
    # plt.show()
    # plt.plot(x,y)
    # plt.show()
    # plt.imshow(img*255)

    plt.figure(figsize=(5,5))
    plt.imshow(scrack)
    plt.show()
    
    # plt.figure(figsize=(10,10))
    # for i in range(9):
    #     plt.subplot(3,3,i+1)
    #     plt.imshow(vol[:,i*100,:])

    # plt.show()


    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.voxels(vol[::100,::100,::100])
    # ax.set_xlabel('Z (build direction)')
    # ax.set_ylabel('Y (image height)')
    # ax.set_zlabel('X (image width)')

    # plt.show()

