
from time import time
import numpy as np
import matplotlib.pyplot as plt

def bisection_search_1d(y,start=0,end=10):
    ''' Find the root of the function f(x) = x^2 - y using the bisection method. 
    '''
    while True:
        mid = (start + end)/2
        print(f"mid: {mid}")
        if abs(f(mid) - y) < 0.0001:
            return mid
        elif f(mid)-y > 0:
            end = mid
        else:
            start = mid

# def bisection_search_2d(f, y,start=(0,0),end=(10,10),epsilon=0.0001):
#     ''' Find the root of the function f(x) = x^2 - y using the bisection method. 
#     '''
#     # start = start*np.ones([10])
#     start = np.array(start)
#     # end = end*np.ones([10])
#     end = np.array(end)

    

#     while True:
#         mid = (start + end)/2
#         midx = mid[0]
#         midy = mid[1]

#         print(f"mid: {mid}")
#         if abs(f(mid) - y) < epsilon:
#             return mid
#         elif (f(midx,start[1])-y) * (f(midx,midy)-y) < 0:
#             end = mid
#         else:
#             start = mid


def bisection_search_2d(f,desired,x_start,x_end,tolerance = 0.001, max_iterations=100):
    while True:
        mid = (x_start + x_end)/2
        print(f"mid: {mid}")
        if abs(f(mid) - y) < 0.0001:
            return mid
        elif f(mid)-y > 0:
            end = mid
        else:
            start = mid


def bisection_search_2d_2var(x_start, x_end, y_start, y_end, tolerance=0.001, max_iterations=100):
    x_mid = (x_start + x_end) / 2
    y_mid = (y_start + y_end) / 2


    for _ in range(max_iterations):

        plot_box(x_start, x_end, y_start, y_end)

        if abs(f(x_mid, y_mid)) < tolerance:
            plt.show()
            return x_mid, y_mid

        A =f(x_mid, y_start) * f(x_mid, y_mid) 
        print(f"A: {A}")
        if  A <= 0:
            y_end = y_mid
        else:
            y_start = y_mid

        B = f(x_start, y_mid) * f(x_mid, y_mid)
        print(f"B: {B}")
        if B <= 0:
            x_end = x_mid
        else:
            x_start = x_mid


        x_mid = (x_start + x_end) / 2
        y_mid = (y_start + y_end) / 2
        print(f"x_mid: {x_mid}, y_mid: {y_mid}")

    plt.show()
    return None

def plot_box(x_start, x_end, y_start, y_end):
    plt.plot([x_start, x_end], [y_start, y_start], color='black')
    plt.plot([x_start, x_end], [y_end, y_end], color='black')
    plt.plot([x_start, x_start], [y_start, y_end], color='black')
    plt.plot([x_end, x_end], [y_start, y_end], color='black')

    
    plt.xlim(0, 16)
    plt.ylim(0, 16)
    plt.axis('equal')

    # plt.show()




def f(x,y):
    return x

if __name__ == "__main__":
    start = time()
    # print(bisection_search_1d(5))
    desired = np.linspace(0,16,100) + np.random.normal(0,1,100)
    plt.plot(desired)
    print(bisection_search_2d_2var(0,16,0,16,tolerance=1))
    print(f"Time: {np.round(time()-start,16)} seconds")
    