"""
Code source : https://github.com/dsgiitr/d2l-pytorch/blob/master/Ch09_Modern_Convolutional_Networks/Residual_Networks_(ResNet).ipynb
"""

import torch 
import torch.nn as nn 
import torch.nn.functional as F 





class d2l_ResNet(nn.Module):
    def __init__(self, num_classes=10):
        super(d2l_ResNet, self).__init__() 


        self.b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                                nn.BatchNorm2d(64),
                                nn.ReLU(),
                                nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        
        self.b2 = nn.Sequential(*self.resnet_block(64, 64, 2, first_block=True))
        self.b3 = nn.Sequential(*self.resnet_block(64, 128, 2))
        self.b4 = nn.Sequential(*self.resnet_block(128, 256, 2))
        self.b5 = nn.Sequential(*self.resnet_block(256, 512, 2))

        self. model = nn.Sequential(
                        self.b1, self.b2, self.b3, self.b4, self.b5,
                        
                        nn.AdaptiveAvgPool2d((1,1)),

                        # Classifier 
                        nn.Flatten(),
                        nn.Linear(in_features=512, out_features=num_classes)       
                        )

    def forward(self, x):
        return self.model(x)





    def resnet_block(self, input_channels, num_channels, num_residuals, first_block=False):
        blk = []

        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.append( Residual_Block(input_channels, num_channels, use_1x1conv=True, strides=2))
            
            else:
                blk.append(Residual_Block(num_channels, num_channels))

        return blk




class Residual_Block(nn.Module):
    def __init__(self, input_channels, num_channels, use_1x1conv=False, strides=1, **kwargs):
        super(Residual_Block, self).__init__(**kwargs)

        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=num_channels, kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)

        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels,num_channels, kernel_size=1, stride=strides)
        else: 
            self.conv3 = None 

        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)
        self.relu = nn.ReLU(inplace=True)

    
    def forward(self, x):
        y = self.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))

        if self.conv3:
            x = self.conv3(x)

        y += x   # residual 
        y = self.relu(y)
        return y






res_blk = Residual_Block(3, 3)

X = torch.randn(size=(4,3,6,6))
y = res_blk(X)
print(y.shape)


res_blk2 = Residual_Block(3, 6, use_1x1conv=True, strides=2)

X = torch.randn(size=(4,3,6,6))
y = res_blk2(X)
print(y.shape)



net = d2l_ResNet()

X = torch.randn(size=(1,1,224,224))

for layer in net.model:
    X=layer(X)
    print(layer.__class__.__name__,'Output shape:\t',X.shape)