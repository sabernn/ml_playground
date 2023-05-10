


import numpy as np
import torch
import matplotlib.pyplot as plt



if __name__ == "__main__":


    x = torch.tensor([1, 2, 3, 4, 5],dtype=torch.float32)

    # W = torch.eye(5)
    W2 = -torch.eye(5)
    W = torch.randn([5,5],dtype=torch.float32)

    y = torch.matmul(W,x)
    y = torch.matmul(W2,y)

    print(y)

    plt.plot(x,y)
    plt.imshow(W)
    plt.show()
    

    

