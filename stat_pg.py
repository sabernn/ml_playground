

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

save = True

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
    height = img_size/20
    width = img_size/4
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
    while count_ul > 1 and count_ur > 1 and count_ll > 1 and count_lr > 1 and count < img_size/2:
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

def crack_volume(img,z_height=2048, shrink_factor_width=10,shrink_factor_length=1):
    # kernel = gaussian_filter([100, 100], sigma=(1,1))
    vol = np.zeros((z_height, img.shape[0], img.shape[1]))
    output = img
    count = 0
    vol[count,:,:] = output
    while np.sum(output) != 0 and count < z_height-1:
        # print(np.sum(output))
        count += 1
        output = np.round(gaussian_filter(output, sigma=(shrink_factor_length,shrink_factor_width))/255)*255
        vol[count,:,:] = output

    return vol



if __name__ == '__main__':
    # crack,x,y,img = generate_crack()
    # crack,x,y,z,vol = generate_crack_3d()

    scrack = crack_surface(img_size=512,
                            center=(256,256))
    # vcrack = crack_volume(scrack,center=(512,512))
    vol = crack_volume(scrack,z_height=1024,shrink_factor_width=20,shrink_factor_length=2)
    # plt.plot(crack)
    # plt.show()
    # plt.plot(x,y)
    # plt.show()
    # plt.imshow(img*255)

    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.imshow(scrack)
    plt.subplot(1,2,2)
    plt.imshow(vol[0])
    plt.show()
    
    plt.figure(figsize=(10,10))
    for i in range(9):
        plt.subplot(3,3,i+1)
        plt.imshow(vol[i])

    plt.show()


    if save:
        np.save('vol.npy',vol)


    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.voxels(vol[::100,::100,::100])
    # ax.set_xlabel('Z (build direction)')
    # ax.set_ylabel('Y (image height)')
    # ax.set_zlabel('X (image width)')

    # plt.show()

