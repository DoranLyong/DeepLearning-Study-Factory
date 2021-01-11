"""
Code source: https://github.com/dsgiitr/d2l-pytorch/blob/master/Ch09_Modern_Convolutional_Networks/Networks_with_Parallel_Concatenations_(GoogLeNet).ipynb
"""

import torch 
import torch.nn as nn 
import torch.nn.functional as F 




class d2l_GoogLeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(d2l_GoogLeNet, self).__init__()

        self.b1 = nn.Sequential(nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
                                nn.ReLU(),
                                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                                )

        self.b2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1),
                                nn.ReLU(),
                                nn.Conv2d(64, 192, kernel_size=3, padding=1),
                                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                                )
        
        self.b3 = nn.Sequential(Inception_Block(192, 64, (96, 128), (16, 32), 32),
                                Inception_Block(256, 128, (128, 192), (32, 96), 64),
                                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                                )

        self.b4 = nn.Sequential(Inception_Block(480, 192, (96, 208), (16, 48), 64),
                                Inception_Block(512, 160, (112, 224), (24, 64), 64),
                                Inception_Block(512, 128, (128, 256), (24, 64), 64),
                                Inception_Block(512, 112, (144, 288), (32, 64), 64),
                                Inception_Block(528, 256, (160, 320), (32, 128), 128),
                                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                                )       

        self.b5 = nn.Sequential(Inception_Block(832, 256, (160, 320), (32, 128), 128),
                                Inception_Block(832, 384, (192, 384), (48, 128), 128),
                                nn.AdaptiveMaxPool2d((1,1)),
                                nn.Flatten()
                                )     

        self.model =  nn.Sequential(self.b1, self.b2, self.b3, self.b4, self.b5, 
                                    nn.Linear(in_features=1024, out_features=num_classes)
                                    )                

    
    def forward(self, x):
        return self.model(x)





class Inception_Block(nn.Module): 
    # c1 - c4 are the number of output channels for each layer in the path
    def __init__(self, in_channels, c1,c2, c3, c4, **kwargs):
        super(Inception_Block, self).__init__(**kwargs)

        # Path 1 is a single 1x1 convolutional layer 
        self.p1_1 = nn.Conv2d(in_channels, c1, kernel_size=1) 

        # Path 2 is a 1x1 Conv_layer followed by a 3x3 Conv_layer 
        self.p2_1 = nn.Conv2d(in_channels, c2[0], kernel_size=1)
        self.p2_2 = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
        
        # Path 3 is a 1x1 Conv_layer followed by a 5x5 Conv_layer 
        self.p3_1 = nn.Conv2d(in_channels, c3[0], kernel_size=1)
        self.p3_2 = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)

        # Path 4 is a 3x3 MaxPool_layer followed by a 1x1 Conv_layer 
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_channels, c4, kernel_size=1)


    def forward(self, x):
        p1 = F.relu(self.p1_1(x))
        p2 = F.relu(self.p2_2(F.relu(self.p2_1(x))))
        p3 = F.relu(self.p3_2(F.relu(self.p3_1(x))))
        p4 = F.relu(self.p4_2(self.p4_1(x)))

        # Concatenate the outputs on the channel dimension 
        return torch.cat((p1, p2, p3, p4), dim=1)






net = d2l_GoogLeNet()

X = torch.randn(size=(1,1,96,96))

for layer in net.model:
    X=layer(X)
    print(layer.__class__.__name__,'Output shape:\t',X.shape)