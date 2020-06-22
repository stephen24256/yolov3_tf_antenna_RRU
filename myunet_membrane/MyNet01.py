import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class ConvolutionBlock(nn.Module):
    def __init__(self,in_channels,out_channels,stride=1):
        super(ConvolutionBlock,self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,3,stride=stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, 3, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )
    def forward(self, x):
        return self.layer(x)

class Conv1_1(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(Conv1_1,self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,1,1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )
    def forward(self,x):
        return self.layer(x)

class Downsimpling(nn.Module):
    def __init__(self,in_channels,out_channels):
        super(Downsimpling,self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,3,2,padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
        )
    def forward(self,x):
        return self.layer(x)

class Upsampling(nn.Module):
    def __init__(self,C1):
        super(Upsampling,self).__init__()
        self.layer0 = nn.Sequential(
            nn.Conv2d(C1,C1//2,1,1),
            nn.BatchNorm2d(C1//2),
            nn.ReLU(True)
        )
    def forward(self,x):
        x = F.interpolate(x,scale_factor=2,mode="nearest")
        return self.layer0(x)

class MyNet(nn.Module):
    def __init__(self):
        super(MyNet,self).__init__()
        self.layer0 = nn.Sequential(
            ConvolutionBlock(1,64),
        )
        self.layer1 = nn.Sequential(
            Downsimpling(64,64),
            ConvolutionBlock(64,128),
        )
        self.layer2 = nn.Sequential(
            Downsimpling(128,128),
            ConvolutionBlock(128,256),
        )
        self.layer3 = nn.Sequential(
            Downsimpling(256,256),
            ConvolutionBlock(256,512),
        )
        self.layer4 = nn.Sequential(
            Downsimpling(512,512),
            ConvolutionBlock(512,1024),
        )
        self.enlayer4 =nn.Sequential(
            ConvolutionBlock(1024,512),
        )
        self.enlayer3 = nn.Sequential(
            ConvolutionBlock(512, 256),
        )
        self.enlayer2 = nn.Sequential(
            ConvolutionBlock(256, 128),
        )
        self.enlayer1 = nn.Sequential(
            ConvolutionBlock(128, 64),
        )
        self.upsampling4 = Upsampling(1024)
        self.upsampling3 = Upsampling(512)
        self.upsampling2 = Upsampling(256)
        self.upsampling1 = Upsampling(128)

        self.con1_1 = Conv1_1(64,1)
    def crop(self,x,y):
        N1, C1, H1, W1 = x.shape
        N2, C2, H2, W2 = y.shape
        return x[:, :, ((H1 - H2) // 2):(H1 - (H1 - H2) // 2), ((W1 - W2) // 2):(W1 - (W1 - W2) // 2)]

    def forward(self,x):
        x0 = self.layer0(x)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        # print(x4.shape)
        x4_1 = self.upsampling4(x4)
        x4_2 = self.crop(x3,x4_1)
        x4_3 = torch.cat((x4_1,x4_2),dim=1)
        x44 = self.enlayer4(x4_3)
        # print("44",x44.shape)
        x3_1 = self.upsampling3(x44)
        x3_2 = self.crop(x2,x3_1)
        x3_3 = torch.cat((x3_1,x3_2),dim=1)
        x33 = self.enlayer3(x3_3)
        # print("33",x33.shape)
        x2_1 = self.upsampling2(x33)
        x2_2 = self.crop(x1, x2_1)
        x2_3 = torch.cat((x2_1,x2_2),dim=1)
        x22 = self.enlayer2(x2_3)
        # print("22",x22.shape)
        x1_1 = self.upsampling1(x22)
        x1_2 = self.crop(x0, x1_1)
        x1_3 = torch.cat((x1_1,x1_2),dim=1)
        x11 = self.enlayer1(x1_3)
        # print("11",x11.shape)
        out = self.con1_1(x11)
        return out

if __name__ == '__main__':
    x = torch.randn(2,1,572,572).cuda()
    net = MyNet().cuda()
    out = net(x)
    print(out.shape)




