

import numpy as np
import matplotlib.pyplot as plt


def generate_crack():
    count = np.random.randint(1, 100)
    print(count)
    img = np.zeros((100, 100))
    # count = 10*np.random.uniform(0, 1)
    x = np.linspace(0, 1, 100)
    y = np.ones(100) + np.random.normal(0, 0.1, 100)
    output = [count]
    for ind in range(len(img[0])):
        count = abs(np.random.normal(count-1, 0.5))
        # count - np.round(np.random.uniform(0, 1)*count,0)
        print(count)
        output.append(count)
        if count == 0:
            break
        # ind += 1
    return output,x,y


if __name__ == '__main__':
    crack,x,y = generate_crack()
    plt.plot(crack)
    plt.show()
    plt.plot(x,y)
    plt.show()

