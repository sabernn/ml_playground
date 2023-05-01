

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
        indy = np.int(y[ind])
        img[indy-int(count/2):indy+int(count/2),ind] = 1

    return output,x,y,img


if __name__ == '__main__':
    crack,x,y,img = generate_crack()
    # plt.plot(crack)
    # plt.show()
    # plt.plot(x,y)
    # plt.show()
    plt.imshow(img*255)
    plt.show()

