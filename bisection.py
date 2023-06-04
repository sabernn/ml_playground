
from time import time
import numpy as np

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

def bisection_search_2d(y,start=(0,0),end=(10,10),epsilon=0.0001):
    ''' Find the root of the function f(x) = x^2 - y using the bisection method. 
    '''
    # start = start*np.ones([10])
    start = np.array(start)
    # end = end*np.ones([10])
    end = np.array(end)
    while True:
        mid = (start + end)/2
        print(f"mid: {mid}")
        if abs(f(mid) - y) < epsilon:
            return mid
        elif f(mid)-y > 0:
            end = mid
        else:
            start = mid


def f(x):
    return (x[0]**2 + 4*x[1]**2)

if __name__ == "__main__":
    start = time()
    # print(bisection_search_1d(5))
    print(bisection_search_2d(6,epsilon=0.001))
    print(f"Time: {np.round(time()-start,16)} seconds")
    