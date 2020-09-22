"""
code source :https://github.com/microsoft/Semantics-Aligned-Representation-Learning-for-Person-Re-identification/blob/master/main.py
"""

import torch 
import torch.nn as nn 
import torch.nn.functional as F



######################
#  Decoder Networks  #
######################

#####  ResNet  #####
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

# Use kernel size 4 to make sure deconv(conv(x)) has the same shape as x
# not working well...
# https://distill.pub/2016/deconv-checkerboard/
def deconv3x3(in_planes, out_planes, stride=1):
    return nn.Sequential(   nn.Upsample(scale_factor=stride, mode='bilinear'),
                            nn.ReflectionPad2d(1),
                            nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1, padding=0),
                        )



# Basic resnet block:
# x ---------------- shortcut ---------------(+) -> plain + x
# \___conv___norm____relu____conv____norm____/
class BasicResBlock(nn.Module):
    def __init__(self, inplanes, norm_layer=nn.BatchNorm2d, activation_layer=nn.LeakyReLU(0.2, True)):
        super(BasicResBlock, self).__init__()

        self.norm_layer = norm_layer
        self.activation_layer = activation_layer
        self.inplanes = inplanes

        layers = [  conv3x3(inplanes, inplanes),
                    norm_layer(inplanes),
                    activation_layer,
                    conv3x3(inplanes, inplanes),
                    norm_layer(inplanes)
                ]
        self.res = nn.Sequential(*layers)

    def forward(self, x):
        return self.res(x) + x



# ResBlock: A classic ResBlock with 2 conv layers and a up/downsample conv layer. (2+1)
# x ---- BasicConvBlock ---- ReLU ---- conv/upconv ----
# If direction is "down", we use nn.Conv2d with stride > 1, getting a smaller image
# If direction is "up", we use nn.ConvTranspose2d with stride > 1, getting a larger image
class ConvResBlock(nn.Module):
    def __init__(self, inplanes, planes, direction, stride=1, norm_layer=nn.BatchNorm2d, activation_layer=nn.LeakyReLU(0.2, True)):
        super(ConvResBlock, self).__init__()
        self.res = BasicResBlock(   inplanes, norm_layer=norm_layer, activation_layer=activation_layer)
        self.activation = activation_layer

        if stride == 1 and inplanes == planes:
            conv = lambda x: x
        else:
            if direction == 'down':
                conv = conv3x3(inplanes, planes, stride=stride)
            elif direction == 'up':
                conv = deconv3x3(inplanes, planes, stride=stride)
            else:
                raise (ValueError('Direction must be either "down" or "up", get %s instead.' % direction))
        self.conv = conv
        self.inplanes = inplanes
        self.planes = planes
        self.stride = stride

    def forward(self, x):
        return self.conv(self.activation(self.res(x)))


#####  Decoder #####
class ConvResDecoder(nn.Module):
    '''
        ConvResDecoder: Use convres block for upsampling
    '''

    def __init__(self):
        super(ConvResDecoder, self).__init__()

        # Xin Jin: this is R-50 inter-channel (2048) with last_stride = 1
        input_channel = 2048
        final_channel = 16 # 16

        # For UNet structure:
        self.embed_layer3 = nn.Sequential(
                                        nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1, stride=1, padding=0),
                                        nn.BatchNorm2d(512),
                                        nn.ReLU(inplace=True)
                                        )
        self.embed_layer2 = nn.Sequential(
                                        nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0),
                                        nn.BatchNorm2d(256),
                                        nn.ReLU(inplace=True)
                                        )
        self.embed_layer1 = nn.Sequential(
                                        nn.Conv2d(in_channels=256, out_channels=64,kernel_size=1, stride=1, padding=0),
                                        nn.BatchNorm2d(64),
                                        nn.ReLU(inplace=True)
                                        )

        self.reduce_dim = nn.Sequential(
                                        nn.Conv2d(input_channel, input_channel//4, kernel_size=1, stride=1, padding=0),
                                        nn.BatchNorm2d(512),
                                        nn.ReLU(inplace=True)
                                        )     # torch.Size([64, 512, 16, 8])

        self.up1 = ConvResBlock(512, 256, direction='up', stride=2, norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU(inplace=True)) # torch.Size([64, 256, 32, 16])
        self.up2 = ConvResBlock(256, 64, direction='up', stride=2, norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU(inplace=True))  # torch.Size([64, 64, 64, 32])
        self.up3 = ConvResBlock(64, 32, direction='up', stride=2, norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU(inplace=True))   # torch.Size([64, 32, 128, 64])
        self.up4 = ConvResBlock(32, 16, direction='up', stride=2, norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU(inplace=True))   # torch.Size([64, 16, 256, 128])

        self.final_layer = nn.Sequential(
                                        nn.ReflectionPad2d(3),
                                        nn.Conv2d(final_channel, 3, kernel_size=7, stride=1, padding=0)  # torch.Size([64, 3, 256, 128])
                                        #nn.Tanh()
                                        )

    def forward(self, x, x_down1, x_down2, x_down3):

        x_reduce_dim = self.reduce_dim(x)          # torch.Size([64, 512, 16, 8])
        embed_layer3 = self.embed_layer3(x_down3)  # torch.Size([64, 512, 16, 8])
        x = self.up1(embed_layer3 + x_reduce_dim)  # torch.Size([64, 256, 32, 16])
        x_sim1 = x
        embed_layer2 = self.embed_layer2(x_down2)  # torch.Size([64, 256, 32, 16])
        x = self.up2(embed_layer2 + x)             # torch.Size([64, 64, 64, 32])
        x_sim2 = x
        embed_layer1 = self.embed_layer1(x_down1)  # torch.Size([64, 64, 64, 32])
        x = self.up3(embed_layer1 + x)             # torch.Size([64, 32, 128, 64])
        x_sim3 = x
        x = self.up4(x)                            # torch.Size([64, 16, 256, 128])
        x_sim4 = x
        x = self.final_layer(x)                    # torch.Size([64, 3, 256, 128])

        # reconstruct the original size, by Jinx:
        x = F.interpolate(x, size=(x.size(2), x.size(3)*2), mode='bilinear', align_corners=True)
        return x, x_sim1, x_sim2, x_sim3, x_sim4




net = ConvResDecoder()

X = torch.randn(size=(64, 512, 16, 8))

"""
for layer in net.model:
    X=layer(X)
    print(layer.__class__.__name__,'Output shape:\t',X.shape)
"""
