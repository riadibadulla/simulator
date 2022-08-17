from numpy import nan
from torch.nn.modules.pooling import AdaptiveAvgPool2d
import torch.nn as nn
import torch.nn.functional as F
from simulator.OpticalConv2d import OpticalConv2d

def opt_conv_block(in_channels, out_channels, k=3, pool_size=0, input_size=10):
  layers = [
            # nn.Conv2d(in_channels, out_channels, kernel_size=k, padding="same"),
            OpticalConv2d(in_channels, out_channels, kernel_size=k, pseudo_negativity=True, input_size=input_size),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)]
  if pool_size !=0:
    layers.append(nn.AdaptiveAvgPool2d(pool_size))
  return nn.Sequential(*layers)

def conv_block(in_channels, out_channels, k=3 ,pool_size=0):
  layers = [nn.Conv2d(in_channels, out_channels, kernel_size=k, padding="same"),
            nn.BatchNorm2d(out_channels),
            # nn.Dropout2d(0.15),
            nn.ReLU(inplace=True)]
  if pool_size !=0:
    layers.append(nn.AdaptiveAvgPool2d(pool_size))
  return nn.Sequential(*layers)


class FATNET_OPTICAL_LRELU(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = opt_conv_block(3, 64, k=3, pool_size=0, input_size=32)
        self.conv2 = opt_conv_block(64, 128, k=3, pool_size=16, input_size=32)
        self.res1 = nn.Sequential(opt_conv_block(128, 128, k=3, pool_size=0, input_size=16), opt_conv_block(128, 128, k=3, pool_size=0, input_size=16))
        self.conv3 = opt_conv_block(128, 164, k=4, pool_size=10, input_size=16)
        self.conv4 = opt_conv_block(164, 82, k=9, pool_size=0)
        self.res2 = nn.Sequential(opt_conv_block(82, 82, k=19, pool_size=0), opt_conv_block(82, 82, k=19, pool_size=0))
        self.conv5 = opt_conv_block(82, 40, k=37, pool_size=0)
        self.classifier = nn.Sequential(OpticalConv2d(40, 1, kernel_size=50, pseudo_negativity=True, input_size=10),
                                        nn.Flatten())

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.res1(x)+x
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.res2(x)+x
        x = self.conv5(x)
        x = self.classifier(x)
        return x

class FATNET_OPTICAL_REDUCED(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = opt_conv_block(3, 64, k=3, pool_size=0, input_size=32)
        self.conv2 = opt_conv_block(64, 128, k=3, pool_size=16, input_size=32)
        self.res1 = nn.Sequential(opt_conv_block(128, 128, k=3, pool_size=0, input_size=16), opt_conv_block(128, 128, k=3, pool_size=0, input_size=16))
        self.conv3 = opt_conv_block(128, 164, k=3, pool_size=10, input_size=16)
        self.conv4 = opt_conv_block(164, 82, k=7, pool_size=0)
        self.res2 = nn.Sequential(opt_conv_block(82, 82, k=13, pool_size=0), opt_conv_block(82, 82, k=13, pool_size=0))
        self.conv5 = opt_conv_block(82, 40, k=27, pool_size=0)
        self.res3 = nn.Sequential(opt_conv_block(40, 40, k=18, pool_size=0), opt_conv_block(40, 40, k=18, pool_size=0))
        self.classifier = nn.Sequential(OpticalConv2d(40, 1, kernel_size=35, pseudo_negativity=True, input_size=10),
                                        nn.Flatten())

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.res1(x)+x
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.res2(x)+x
        x = self.conv5(x)
        x = self.res3(x)+x
        x = self.classifier(x)
        return x

class FNETNET(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = conv_block(3,64, k=3 ,pool_size=32)
        self.conv2 = conv_block(64,128, k=3, pool_size=16)
        self.res1 = nn.Sequential(conv_block(128,128, k=3,pool_size=0), conv_block(128,128,k=3,pool_size=0))
        self.conv3 = conv_block(128,164, k=4, pool_size=10)
        self.conv4 = conv_block(164,82, k=9, pool_size=0)
        self.res2 = nn.Sequential(conv_block(82,82, k=19,pool_size=0), conv_block(82,82, k=19 ,pool_size=0))
        self.conv5 = conv_block(82,40, k=37, pool_size=0)
        self.res3 = nn.Sequential(conv_block(40,40,k=25, pool_size=0), conv_block(40,40,k=25, pool_size=0))
        self.classifier = nn.Sequential(nn.Conv2d(40, 1, kernel_size=40, padding="same"),
                                        nn.Flatten())

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.res1(x)+x
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.res2(x)+x
        x = self.conv5(x)
        x = self.res3(x)+x
        x = self.classifier(x)
        return x

class FATNET_OPTICAL(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = opt_conv_block(3, 64, k=3, pool_size=0, input_size=32)
        self.conv2 = opt_conv_block(64, 128, k=3, pool_size=16, input_size=32)
        self.res1 = nn.Sequential(opt_conv_block(128, 128, k=3, pool_size=0, input_size=16), opt_conv_block(128, 128, k=3, pool_size=0, input_size=16))
        self.conv3 = opt_conv_block(128, 164, k=4, pool_size=10, input_size=16)
        self.conv4 = opt_conv_block(164, 82, k=9, pool_size=0)
        self.res2 = nn.Sequential(opt_conv_block(82, 82, k=19, pool_size=0), opt_conv_block(82, 82, k=19, pool_size=0))
        self.conv5 = opt_conv_block(82, 40, k=37, pool_size=0)
        self.res3 = nn.Sequential(opt_conv_block(40, 40, k=25, pool_size=0), opt_conv_block(40, 40, k=25, pool_size=0))
        self.classifier = nn.Sequential(OpticalConv2d(40, 1, kernel_size=50, pseudo_negativity=True, input_size=10),
                                        nn.Flatten())

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.res1(x)+x
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.res2(x)+x
        x = self.conv5(x)
        x = self.res3(x)+x
        x = self.classifier(x)
        return x

class OPTICAL_ORIGINAL(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = opt_conv_block(3,64, k=3, pool_size=0, input_size=32)
        self.conv2 = opt_conv_block(64,128,k=3,pool_size=16, input_size=32)
        self.res1 = nn.Sequential(opt_conv_block(128,128, k=3, pool_size=0, input_size=16), opt_conv_block(128,128, k=3, pool_size=0,input_size=16))
        self.conv3 = opt_conv_block(128,256, k=3, pool_size=8,input_size=16)
        self.conv4 = opt_conv_block(256,512,k=3, pool_size=4,input_size=8)
        self.res2 = nn.Sequential(opt_conv_block(512,512, k=3, pool_size=0,input_size=4), opt_conv_block(512,512, k=3, pool_size=0,input_size=4))
        self.conv5 = opt_conv_block(512,1000,k=3, pool_size=2,input_size=4)
        self.res3 = nn.Sequential(opt_conv_block(1000,1000, k=3, pool_size=0,input_size=2), opt_conv_block(1000,1000, k=3, pool_size=0,input_size=2))
        self.pool = nn.MaxPool2d(2)
        self.classifier = nn.Sequential(nn.Flatten(), 
                                          nn.Linear(1000, 100))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.res1(x)+x
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.res2(x)+x
        x = self.conv5(x)
        x = self.res3(x)+x
        x = self.pool(x)
        x = self.classifier(x)
        return x

class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = conv_block(3,64,k=7,pool_size=16)
        self.res1 = nn.Sequential(conv_block(64,64), conv_block(64,64))
        self.res2 = nn.Sequential(conv_block(64,64), conv_block(64,64))

        self.conv2 = nn.Sequential(conv_block(64,128,pool_size=8), conv_block(128,128))
        self.res3 = nn.Sequential(conv_block(128,128), conv_block(128,128))

        self.conv3 = nn.Sequential(conv_block(128,256,pool_size=4), conv_block(256,256))
        self.res4 = nn.Sequential(conv_block(256,256), conv_block(256,256))

        self.conv4 = nn.Sequential(conv_block(256,512,pool_size=2), conv_block(512,512))
        self.res5 = nn.Sequential(conv_block(512,512), conv_block(512,512))
        self.pool = nn.MaxPool2d(2)
        self.classifier = nn.Sequential(nn.Flatten(),
                                        nn.Dropout(p=0.2, inplace=False),
                                        nn.Linear(512, 100))

    def forward(self, x):
        x = self.conv1(x)
        x = self.res1(x)+x
        x = self.res2(x)+x
        x = self.conv2(x)
        x = self.res3(x)+x
        x = self.conv3(x)
        x = self.res4(x)+x
        x = self.conv4(x)
        x = self.res5(x)+x
        x = self.pool(x)
        x = self.classifier(x)
        return x

class ResFATNET(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = conv_block(3,64,k=7,pool_size=16)
        self.res1 = nn.Sequential(conv_block(64,64), conv_block(64,64))
        self.res2 = nn.Sequential(conv_block(64,64,), conv_block(64,64))

        self.conv2 = nn.Sequential(conv_block(64,82,k=4,pool_size=10), conv_block(82,82,k=5))
        self.res3 = nn.Sequential(conv_block(82,82,k=5), conv_block(82,82,k=5))

        self.conv3 = nn.Sequential(conv_block(82,41,k=9), conv_block(41,41,k=19))
        self.res4 = nn.Sequential(conv_block(41,41,k=19), conv_block(41,41,k=19))

        self.conv4 = nn.Sequential(conv_block(41,21,k=37), conv_block(21,21,k=73))
        self.res5 = nn.Sequential(conv_block(21,21,k=73), conv_block(21,21,k=73))
        self.classifier = nn.Sequential(
                                        nn.Dropout(p=0.2, inplace=False),
                                        nn.Conv2d(21, 1, kernel_size=49, padding="same"),
                                        nn.Flatten())

    def forward(self, x):
        x = self.conv1(x)
        x = self.res1(x)+x
        x = self.res2(x)+x
        x = self.conv2(x)
        x = self.res3(x)+x
        x = self.conv3(x)
        x = self.res4(x)+x
        x = self.conv4(x)
        x = self.res5(x)+x
        x = self.classifier(x)
        return x

class ResFATNETSMALLKERNEL(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = conv_block(3,64,k=7,pool_size=16)
        self.res1 = nn.Sequential(conv_block(64,64), conv_block(64,64))
        self.res2 = nn.Sequential(conv_block(64,64,), conv_block(64,64))

        self.conv2 = nn.Sequential(conv_block(64,82,k=4,pool_size=10), conv_block(82,82,k=5))
        self.res3 = nn.Sequential(conv_block(82,82,k=5), conv_block(82,82,k=5))

        self.conv3 = nn.Sequential(conv_block(82,78,k=7), conv_block(78,78,k=10))
        self.res4 = nn.Sequential(conv_block(78,78,k=10), conv_block(78,78,k=10))

        self.conv4 = nn.Sequential(conv_block(78,151,k=10), conv_block(151,155,k=10))
        self.res5 = nn.Sequential(conv_block(155,151,k=10), conv_block(151,155,k=10))
        self.classifier = nn.Sequential(
                                        nn.Dropout(p=0.2, inplace=False),
                                        nn.Conv2d(155, 1, kernel_size=10, padding="same"),
                                        nn.Flatten())

    def forward(self, x):
        x = self.conv1(x)
        x = self.res1(x)+x
        x = self.res2(x)+x
        x = self.conv2(x)
        x = self.res3(x)+x
        x = self.conv3(x)
        x = self.res4(x)+x
        x = self.conv4(x)
        x = self.res5(x)+x
        x = self.classifier(x)
        return x

class ResFATNETSMALLKERNEL2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = conv_block(3,64,k=7,pool_size=16)
        self.res1 = nn.Sequential(conv_block(64,64), conv_block(64,64))
        self.res2 = nn.Sequential(conv_block(64,64,), conv_block(64,64))

        self.conv2 = nn.Sequential(conv_block(64,38,k=10,pool_size=10), conv_block(38,38,k=10))
        self.res3 = nn.Sequential(conv_block(38,38,k=10), conv_block(38,38,k=10))

        self.conv3 = nn.Sequential(conv_block(38,76,k=10), conv_block(76,76,k=10))
        self.res4 = nn.Sequential(conv_block(76,76,k=10), conv_block(76,76,k=10))

        self.conv4 = nn.Sequential(conv_block(76,153,k=10), conv_block(153,153,k=10))
        self.res5 = nn.Sequential(conv_block(153,153,k=10), conv_block(153,153,k=10))
        self.classifier = nn.Sequential(
                                        nn.Dropout(p=0.2, inplace=False),
                                        nn.Conv2d(153, 1, kernel_size=10, padding="same"),
                                        nn.Flatten())

    def forward(self, x):
        x = self.conv1(x)
        x = self.res1(x)+x
        x = self.res2(x)+x
        x = self.conv2(x)
        x = self.res3(x)+x
        x = self.conv3(x)
        x = self.res4(x)+x
        x = self.conv4(x)
        x = self.res5(x)+x
        x = self.classifier(x)
        return x

class Optical_ResFATNET(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = opt_conv_block(3,64,k=7,pool_size=16, input_size=32)
        self.res1 = nn.Sequential(opt_conv_block(64,64, input_size=16), opt_conv_block(64,64, input_size=16))
        self.res2 = nn.Sequential(opt_conv_block(64,64, input_size=16), opt_conv_block(64,64, input_size=16))

        self.conv2 = nn.Sequential(opt_conv_block(64,82,k=4,pool_size=10, input_size=16), opt_conv_block(82,82,k=5))
        self.res3 = nn.Sequential(opt_conv_block(82,82,k=5), opt_conv_block(82,82,k=5))

        self.conv3 = nn.Sequential(opt_conv_block(82,78,k=7), opt_conv_block(78,78,k=10))
        self.res4 = nn.Sequential(opt_conv_block(78,78,k=10), opt_conv_block(78,78,k=10))

        self.conv4 = nn.Sequential(opt_conv_block(78,151,k=10), opt_conv_block(151,155,k=10))
        self.res5 = nn.Sequential(opt_conv_block(155,151,k=10), opt_conv_block(151,155,k=10))
        self.classifier = nn.Sequential(nn.Dropout(p=0.2, inplace=False),
                                        OpticalConv2d(155, 1, kernel_size=10, pseudo_negativity=True, input_size=10),
                                        nn.Flatten())

    def forward(self, x):
        x = self.conv1(x)
        x = self.res1(x)+x
        x = self.res2(x)+x
        x = self.conv2(x)
        x = self.res3(x)+x
        x = self.conv3(x)
        x = self.res4(x)+x
        x = self.conv4(x)
        x = self.res5(x)+x
        x = self.classifier(x)
        return x       

def get_model(name):
    if name =="FATNET_OPTICAL":
        return FATNET_OPTICAL()
    elif name =="FARNET":
        return FNETNET()
    elif name =="FATNET_OPTICAL_REDUCED":
        return FATNET_OPTICAL_REDUCED()
    elif name=="FATNET_OPTICAL_LRELU":
        return FATNET_OPTICAL_LRELU()
    elif name=="OPTICAL_ORIGINAL":
        return OPTICAL_ORIGINAL()
    elif name=="RESNET":
        return ResNet()
    elif name=="RESFATNET":
        return ResFATNET()
    elif name=="OPTICALRESFATNET":
        return Optical_ResFATNET()
    elif name=="ResFATNETSMALLKERNEL":
        return ResFATNETSMALLKERNEL()
    elif name=="ResFATNETSMALLKERNEL2":
        return ResFATNETSMALLKERNEL2()
