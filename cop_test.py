
'''
Write a function that takes a number and adds it to all the numbers before it.
'''
def add_to_all(num):
    total = 0
    for i in range(num):
        total += i
    return total

import torch
import torch.nn as nn


''' 
Write a class for U-Net architecture.
'''
class UNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        return x
    
    def train(self):
        pass

    def test(self):
        pass



