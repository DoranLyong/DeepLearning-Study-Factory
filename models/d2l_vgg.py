"""
Code source : https://github.com/dsgiitr/d2l-pytorch/blob/master/Ch09_Modern_Convolutional_Networks/VGG.ipynb
"""

import torch 
import torch.nn as nn 
import torch.nn.functional as F 


conv_arch = ((1, 64), (1, 128), (2, 256), (2, 512), (2, 512))



class d2l_VGG(nn.Module):
    def __init__(self, num_classes=10, conv_arch=conv_arch ):
        super(d2l_VGG, self).__init__()

        # The convolutional layer part 
        conv_layers = [] 
        in_channels = 1

        for (num_convs, out_channels) in conv_arch:
            conv_layers.append(self.vgg_block(num_convs, in_channels, out_channels))
            in_channels = out_channels

        
        self.model = nn.Sequential(
                                    *conv_layers,

                                    # FC-layer part
                                    nn.Flatten(),
                                    nn.Linear(in_features=512*7*7, out_features=4096),
                                    nn.ReLU(),
                                    nn.Dropout(0.5),
                                    nn.Linear(4096, 4096),
                                    nn.ReLU(),
                                    nn.Dropout(0.5),
                                    nn.Linear(in_features=4096, out_features=num_classes),                                    
                                    )


    def forward(self, x):
        return self.model(x)



    def vgg_block(self, num_convs, in_channels, out_channels): 
        layers = [] 

        for _ in range(num_convs):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.ReLU())            
            in_channels = out_channels

        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        
        blk = nn.Sequential(*layers)

        return blk




net = d2l_VGG()

X = torch.randn(size=(1,1,224,224))

for layer in net.model:
    X=layer(X)
    print(layer.__class__.__name__,'Output shape:\t',X.shape)