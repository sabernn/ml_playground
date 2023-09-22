

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import cv2
import patchify as pf
from albumentations import (
    HorizontalFlip,
    VerticalFlip,
    Compose,
    CLAHE,
    RandomRotate90,
    Rotate,
    IAAPiecewiseAffine,
    IAAPerspective,
    RandomContrast,#limit=0.2
    RandomBrightness,#limit=0.2
    GaussNoise,#var_limit=50
    Normalize#Default:mean,std of ImageNet 2012 {mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]}
)


save = False

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
    d = 1024
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

def surface_crack(img_size: int, crack_count: int = 10, min_crack_thickness: int = 1, max_crack_thickness: int = 10, hor: bool = True):
    # img_size = 1024

    # height = img_size/20
    # width = img_size/4
    
    # count_ul = height
    # count_ur = height
    # count_ll = height
    # count_lr = height
    img = np.zeros((img_size, img_size))
    
    # crack_count = np.random.randint(1, 10)
    # print(f"crack count: {crack_count}")
    centers = np.random.randint(0, img_size, size=(2,crack_count))
    for i in range(crack_count):
        

        cx = centers[0, i]
        cyl = centers[1, i]
        cyr = centers[1, i]
        smoothness_factor = 1 # higher = smoother
        count = 0
        crack_thickness = np.random.randint(min_crack_thickness, max_crack_thickness)
        count_l = crack_thickness
        count_r = crack_thickness
        # while count_ul > 1 and count_ur > 1 and count_ll > 1 and count_lr > 1 and count < img_size/2:
        hor = np.random.randint(0, 2)
        while count_l > 1 and count_r > 1 and count < img_size/2:
            # random asymmetrical decay of crack width
            if count%smoothness_factor == 0:
                count_l *= (1-abs(np.random.normal(0, 0.01*smoothness_factor)))
                count_r *= (1-abs(np.random.normal(0, 0.01*smoothness_factor)))
                # count_ul *= (1-abs(np.random.normal(0, 0.01*smoothness_factor)))
                # count_ur *= (1-abs(np.random.normal(0, 0.01*smoothness_factor)))
                # count_ll *= (1-abs(np.random.normal(0, 0.01*smoothness_factor)))
                # count_lr *= (1-abs(np.random.normal(0, 0.01*smoothness_factor)))
                cyl += np.random.normal(0, 1)
                cyr += np.random.normal(0, 1)
            # img[int(cyl-count_ll):int(cyl+count_ul),cx-count:cx-count+1] = 255
            # img[int(cyr-count_lr):int(cyr+count_ur),cx+count:cx+count+1] = 255
            if hor:
                img[int(cyl-count_l/2):int(cyl+count_l/2),cx-count:cx-count+1] = 255
                img[int(cyr-count_r/2):int(cyr+count_r/2),cx+count:cx+count+1] = 255
            else:
                img[cx-count:cx-count+1, int(cyl-count_l/2):int(cyl+count_l/2)] = 255
                img[cx+count:cx+count+1, int(cyr-count_r/2):int(cyr+count_r/2)] = 255


            count += 1

    return img

def gauss_2d(img_size, center, sigma):
    x, y = np.meshgrid(np.linspace(0, img_size, img_size), np.linspace(0, img_size, img_size))
    d = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    mu, sigma = 0, sigma
    gauss = np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) )
    return gauss

def volume_crack(base_size: int, height_size: int):
    
    # count_ul = height
    # count_ur = height
    # count_ll = height
    # count_lr = height
    vol = np.zeros((height_size, base_size, base_size))
    
    # crack_count = np.random.randint(1, 10)
    crack_count = 1

    print(f"crack count: {crack_count}")
    centers = np.random.randint(0, base_size, size=(3,crack_count))
    for i in range(crack_count):
        # x, y = np.meshgrid(np.linspace(0, base_size, base_size), np.linspace(0, base_size, base_size))
        # d = np.sqrt((x - centers[0, i])**2 + (y - centers[1, i])**2)
        # mu, sigma = 0, 10
        # gauss = np.exp(-( (d-mu)**2 / ( 2.0 * sigma**2 ) ) )
        height = np.random.randint(1, base_size/10)
        # height = 5
        # gauss = gauss_2d(base_size, (centers[0, i], centers[1, i]), 100)
        # gauss_crack = height*gauss + gauss*abs(np.random.normal(0, 10, size=(base_size, base_size)))
        # indvol = gauss_crack > 0.5

        # indvol_value = np.zeros((height_size, base_size, base_size))
        # indvol_value[:, :, centers[2, i]] = gauss_crack[indvol]

        # vol[:, :, centers[2, i]] = gauss_crack

        cx = centers[0, i]
        cy = centers[1, i]
        cz_ul = centers[2, i]
        cz_ur = centers[2, i]
        cz_ll = centers[2, i]
        cz_lr = centers[2, i]
        smoothness_factor = 1 # higher = smoother
        count = 0
        
        count_ul = height
        count_ur = height
        count_ll = height
        count_lr = height
        while count_ul > 1 and count_ur > 1 and count_ll > 1 and count_lr > 1 and count < base_size/2:
        # while count_l > 1 and count_r > 1 and count < img_size/2:
            # random asymmetrical decay of crack width
            if count%smoothness_factor == 0:
                # count_l *= (1-abs(np.random.normal(0, 0.01*smoothness_factor)))
                # count_r *= (1-abs(np.random.normal(0, 0.01*smoothness_factor)))
                count_ul *= (1-abs(np.random.normal(0, 0.0001*smoothness_factor)))
                count_ur *= (1-abs(np.random.normal(0, 0.0001*smoothness_factor)))
                count_ll *= (1-abs(np.random.normal(0, 0.0001*smoothness_factor)))
                count_lr *= (1-abs(np.random.normal(0, 0.0001*smoothness_factor)))
                cz_ul += np.random.normal(0, 1)
                cz_ur += np.random.normal(0, 1)
                cz_ll += np.random.normal(0, 1)
                cz_lr += np.random.normal(0, 1)
            # img[int(cyl-count_ll):int(cyl+count_ul),cx-count:cx-count+1] = 255
            # img[int(cyr-count_lr):int(cyr+count_ur),cx+count:cx+count+1] = 255

            # vol[int(cz_ul-count_ul/2):int(cz_ul+count_ul/2),cy-count:cy-count+1,cx-count:cx-count+1] = 255
            # vol[int(cz_ur-count_ur/2):int(cz_ur+count_ur/2),cy-count:cy-count+1,cx+count:cx+count+1] = 255
            # vol[int(cz_ll-count_ll/2):int(cz_ll+count_ll/2),cy+count:cy+count+1,cx-count:cx-count+1] = 255 
            # vol[int(cz_lr-count_lr/2):int(cz_lr+count_lr/2),cy+count:cy+count+1,cx+count:cx+count+1] = 255
            vol[int(cz_ul-count_ul/2):int(cz_ul+count_ul/2),cy:cy-count+1,cx:cx-count+1] = 255
            vol[int(cz_ur-count_ur/2):int(cz_ur+count_ur/2),cy:cy-count+1,cx:cx+count+1] = 255
            vol[int(cz_ll-count_ll/2):int(cz_ll+count_ll/2),cy:cy+count+1,cx:cx-count+1] = 255 
            vol[int(cz_lr-count_lr/2):int(cz_lr+count_lr/2),cy:cy+count+1,cx:cx+count+1] = 255

            # np.random.normal([0,0],[1,1])

            # vol[int(cyr-count_r/2):int(cyr+count_r/2),cx+count:cx+count+1] = 255
            count += 1
        print(f"crack volume: {vol.sum()/255}")

    print(f"crack volume fraction: {vol.sum()/(255*base_size*base_size*height_size)}")
    return vol



def volume_crack_from_surface(img,z_height=1024, shrink_factor_width=10,shrink_factor_length=1):
    # kernel = gaussian_filter([100, 100], sigma=(1,1))
    kernel = np.ones((shrink_factor_width,shrink_factor_length),np.uint8)
    vol = np.zeros((z_height, img.shape[0], img.shape[1]))
    output = img
    count = 0
    vol[count,:,:] = output
    # depth_factor = 10 # higher = deeper
    depth_factor = np.random.randint(10, 100)
    print(f"depth factor: {depth_factor}")
    while np.sum(output) != 0 and count < z_height-1:
        # print(np.sum(output))
        count += 1
        # output = np.round(gaussian_filter(output, sigma=(shrink_factor_length,shrink_factor_width))/255)*255
        # output = np.round(cv2.erode(output, kernel)/255)*255
        if count%depth_factor == 0:
            output = cv2.erode(output, kernel)
        # output = cv2.dilate(output, kernel,iterations=1)
        # print(np.sum(output))
        vol[count,:,:] = output

    return vol


def generate_mask(params, save_mask=True):
    count = 0
    for i in range(params['image_count']):
        while count<params['mask_count']:
            img = surface_crack(img_size=params['image_size'],
                                crack_count=np.random.randint(params['min_crack_count'], params['max_crack_count']),
                                min_crack_thickness=params['min_crack_thickness'],
                                max_crack_thickness=params['max_crack_thickness'])
            masks = pf.patchify(img, (params['mask_size'], params['mask_size']), step=512)
            masks = masks.reshape(-1, params['mask_size'], params['mask_size'])
            # l = masks.shape[0]*masks.shape[1]
            for mask in masks:
                
                if mask.sum() > 0:
                    aug = Compose([IAAPiecewiseAffine(scale=(0.09, 0.13), nb_rows=4, nb_cols=4, order=1, cval=0, mode='constant', always_apply=False, p=1),Rotate(limit=30, p=0.5)], p=1)
                    mask = aug(image=mask)['image']
                    if save_mask:
                        cv2.imwrite(f"trainA/{str(count).zfill(5)}.png", mask)
                    count += 1
                    # print(f"mask {count} generated")
            # count += (masks.shape[0]*masks.shape[1])



if __name__ == '__main__':
    # crack,x,y,img = generate_crack()
    # crack,x,y,z,vol = generate_crack_3d()

    params = {'image_size': 1024,
                'image_count': 1000,
                'mask_size': 512,
                'mask_count': 1600,
                'min_crack_count': 10,
                'max_crack_count': 20,
                'min_crack_thickness': 1,
                'max_crack_thickness': 20,

                }
    
    generate_mask(params)

    # random_values = {'crack_count': np.random.randint(10, 20),
    #                     'min_crack_thickness': 1,
    #                     'max_crack_thickness': 10,
    #                     'crack_width': np.random.randint(1, 10),
    #                     'crack_smoothness': np.random.randint(1, 10),
    #                     'crack_depth': np.random.randint(1, 10),
    #                     'crack_depth_factor': np.random.randint(1, 10),
    #                     'crack_shrink_factor_width': np.random.randint(1, 10),
    #                     'crack_shrink_factor_length': np.random.randint(1, 10)}


    # vol = volume_crack(base_size=1024, height_size=1024)
    # print(random_values)
    # scrack = surface_crack(img_size=1024,
    #                         crack_count=random_values['crack_count'],
    #                         min_crack_thickness=random_values['min_crack_thickness'],
    #                         max_crack_thickness=random_values['max_crack_thickness'])
    # # vcrack = crack_volume(scrack,center=(512,512))
    # # vol = crack_volume(scrack,z_height=1024,shrink_factor_width=np.random.randint(1,2),shrink_factor_length=np.random.randint(1,2))
    # vol = volume_crack_from_surface(scrack,z_height=1024,shrink_factor_width=2,shrink_factor_length=2)
    # plt.plot(crack)
    # plt.show()
    # plt.plot(x,y)
    # plt.show()
    # plt.imshow(img*255)

    # plt.figure(figsize=(10,5))
    # plt.subplot(1,2,1)
    # plt.imshow(scrack)
    # plt.subplot(1,2,2)
    # plt.imshow(vol[0])
    # plt.show()
    
    # plt.figure(figsize=(10,10))
    # for i in range(9):
    #     plt.subplot(3,3,i+1)
    #     plt.imshow(vol[10*i])

    # plt.show()

    # plt.figure(figsize=(10,10))
    # for i in range(9):
    #     plt.subplot(3,3,i+1)
    #     plt.imshow(vol[:,10*i,:])

    # plt.show()


    if save:
        np.save('vol.npy',vol)


    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.voxels(vol[::100,::100,::100])
    # ax.set_xlabel('Z (build direction)')
    # ax.set_ylabel('Y (image height)')
    # ax.set_zlabel('X (image width)')

    # plt.show()

