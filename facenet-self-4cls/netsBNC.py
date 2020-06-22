import torch
import torch.nn as nn

class PNet(nn.Module):

    def __init__(self):
        super(PNet, self).__init__()

        self.pre_layer1 = nn.Sequential(
            nn.Conv2d(3, 10, kernel_size=3, stride=1),  # 10*10*3
            nn.BatchNorm2d(10),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2,padding=0),  # 5*5*10
            nn.Conv2d(10, 16, kernel_size=3, stride=1),  # 3*3*16
            nn.BatchNorm2d(16),
            nn.PReLU(),)
        self.pre_layer2= nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1),  # 1*1*32
            nn.BatchNorm2d(32),
            nn.PReLU()
        )
        self.conv4_1 = nn.Conv2d(32, 1, kernel_size=1, stride=1)
        self.conv4_2 = nn.Conv2d(32, 4, kernel_size=1, stride=1)

    def channel_shuffle(self, x, *channels):
        """
        自定义通道数目
        :return:
        """
        # print("x",x.shape)
        N, C, H, W = x.shape
        indices = list(range(len(channels), 0, -1))
        # print("14",N,C,H,W,indices)
        x = x.reshape(N, *channels, H, W)
        x = x.permute(0, *indices, x.dim() - 2, x.dim() - 1)
        x = x.reshape(N, C, H, W)
        return x

    def forward(self, x):
        x = self.pre_layer1(x)
        x = self.channel_shuffle(x,2,2,2,2)
        x = self.pre_layer2(x)
        cls = torch.sigmoid(self.conv4_1(x))
        offset = self.conv4_2(x)
        return cls, offset

class RNet(nn.Module):
    def __init__(self):
        super(RNet, self).__init__()
        self.pre_layer1 = nn.Sequential(
            nn.Conv2d(3, 28, kernel_size=3, stride=1),  # 22*22*28
            nn.BatchNorm2d(28),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2,padding=1),  # 11*11*28
            nn.Conv2d(28, 48, kernel_size=3, stride=1),  # 9*9*48
            nn.BatchNorm2d(48),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),)  # 4*4*48

        self.pre_layer2 = nn.Sequential(
            nn.Conv2d(48, 64, kernel_size=2, stride=1),  # 3*3*64
            nn.BatchNorm2d(64),
            nn.PReLU()
        )
        self.conv4 = nn.Linear(64 * 3 * 3, 128)
        self.prelu4 = nn.PReLU()
        self.conv5_1 = nn.Linear(128, 1)
        self.conv5_2 = nn.Linear(128, 4)
    def channel_shuffle(self, x, *channels):
        """
        自定义通道数目
        :return:
        """
        # print("x",x.shape)
        N, C, H, W = x.shape
        indices = list(range(len(channels), 0, -1))
        # print("14",N,C,H,W,indices)
        x = x.reshape(N, *channels, H, W)
        x = x.permute(0, *indices, x.dim() - 2, x.dim() - 1)
        x = x.reshape(N, C, H, W)
        return x

    def forward(self, x):
        x = self.pre_layer1(x)
        x = self.channel_shuffle(x,2,3,2,4)
        x = self.pre_layer2(x)
        x = x.view(x.size(0), -1)
        x = self.conv4(x)
        x = self.prelu4(x)
        cls = torch.sigmoid(self.conv5_1(x))
        offset = self.conv5_2(x)
        return cls, offset
#
#
class ONet(nn.Module):
    def __init__(self):
        super(ONet, self).__init__()
        self.pre_layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1),  # 46*46*32
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2,padding=1),  # 23*23*32
            nn.Conv2d(32, 64, kernel_size=3, stride=1),  # 21*21*64
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 10*10*64
            nn.Conv2d(64, 64, kernel_size=3, stride=1),  # 8*8*64
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),) # 4*4*64

        self.pre_layer2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=2, stride=1),  # 3*3*128
            nn.BatchNorm2d(128),
            nn.PReLU()
        )
        self.conv5 = nn.Linear(128 * 3 * 3, 256)
        self.prelu5 = nn.PReLU()
        self.conv6_1 = nn.Linear(256, 1)
        self.conv6_2 = nn.Linear(256, 4)

    def channel_shuffle(self, x, *channels):
        """
        自定义通道数目
        :return:
        """
        # print("x",x.shape)
        N, C, H, W = x.shape
        indices = list(range(len(channels), 0, -1))
        # print("14",N,C,H,W,indices)
        x = x.reshape(N, *channels, H, W)
        x = x.permute(0, *indices, x.dim() - 2, x.dim() - 1)
        x = x.reshape(N, C, H, W)
        return x

    def forward(self, x):
        x = self.pre_layer1(x)
        x = self.channel_shuffle(x,2,4,4,2)
        x = self.pre_layer2(x)
        x = x.view(x.size(0), -1)
        x = self.conv5(x)
        x = self.prelu5(x)
        cls = torch.sigmoid(self.conv6_1(x))
        offset = self.conv6_2(x)
        return cls, offset


if __name__ == '__main__':
    x = torch.randn(2,3,12,12)
    net = PNet()
    out1 ,out2 = net(x)
    print(out1.shape,out2.shape)