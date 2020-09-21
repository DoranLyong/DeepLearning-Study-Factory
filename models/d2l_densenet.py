"""
Code source: https://github.com/dsgiitr/d2l-pytorch/blob/master/Ch09_Modern_Convolutional_Networks/Densely_Connected_Networks_(DenseNet).ipynb
"""

import torch 
import torch.nn as nn
import torch.nn.functional as F 






class d2l_DenseNet(nn.Module):
    def __init__(self, num_classes=10):
        super(d2l_DenseNet, self).__init__()







    def forward(self, x): 
        pass 





def transition_block(input_channels, num_channels):
    layers = []
    layers.append(nn.BatchNorm2d(input_channels))
    layers.append(nn.ReLU())
    layers.append(nn.Conv2d(input_channels, num_channels, kernel_size=1))
    layers.append(nn.AvgPool2d(kernel_size=2, stride=2))
    blk = nn.Sequential(*layers)
    return blk



def conv_block(input_channels, num_channels):
    layers = []
    layers.append(nn.BatchNorm2d(input_channels))
    layers.append(nn.ReLU())
    layers.append(nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1))
    blk = nn.Sequential(*layers)
    return blk


class DenseBlock(nn.Module):
    def __init__(self, num_convs, input_channels, num_channels, **kwargs):
        super(DenseBlock, self).__init__(**kwargs)
        layer = []

        for i in range(num_convs):
            layer.append(conv_block((num_channels * i + input_channels), num_channels))
        self.net = nn.Sequential(*layer)

    def forward(self, X):

        for blk in self.net:
            Y = blk(X)
            # Concatenate the input and output of each block on the channel dimension
            X = torch.cat((X, Y), dim=1)
        return X



blk = DenseBlock(2, 3, 10)
X = torch.randn(4, 3, 8, 8)
Y = blk(X)
print(Y.shape)


blk = transition_block(23, 10)
X = torch.randn(4, 3, 8, 8)
print(blk(Y).shape)