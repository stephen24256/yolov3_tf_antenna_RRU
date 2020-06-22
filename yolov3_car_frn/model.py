import torch
import torch.nn.functional as F
from carafe import Carafe
from FRN import FilterResponseNormNd

# #定义上采样层，邻近插值
# class UpsampleLayer(torch.nn.Module):
#     def __init__(self):
#         super(UpsampleLayer, self).__init__()
#
#     def forward(self, x):
#         # return F.interpolate(x, scale_factor=2, mode='nearest')  #
#         return F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)

#定义卷积层
class ConvolutionalLayer(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias=False):
        super(ConvolutionalLayer, self).__init__()

        self.sub_module = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
            FilterResponseNormNd(4, out_channels),
        )

    def forward(self, x):
        return self.sub_module(x)

#定义残差结构
class ResidualLayer(torch.nn.Module):

    def __init__(self, in_channels):
        super(ResidualLayer, self).__init__()

        self.sub_module = torch.nn.Sequential(
            ConvolutionalLayer(in_channels, in_channels // 2, 1, 1, 0),  # N C//2 H W
            ConvolutionalLayer(in_channels // 2, in_channels, 3, 1, 1),  # N C H W
        )

    def forward(self, x):
        return x + self.sub_module(x)   # N C H W + N C H W

#定义下采样层
class DownsamplingLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownsamplingLayer, self).__init__()
        self.sub_module = torch.nn.Sequential(
            ConvolutionalLayer(in_channels, out_channels, 3, 2, 1)
        )

    def forward(self, x):
        return self.sub_module(x)

#定义卷积块
class ConvolutionalSet(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvolutionalSet, self).__init__()

        self.sub_module = torch.nn.Sequential(
            ConvolutionalLayer(in_channels, out_channels, 1, 1, 0),
            ConvolutionalLayer(out_channels, in_channels, 3, 1, 1),

            ConvolutionalLayer(in_channels, out_channels, 1, 1, 0),
            ConvolutionalLayer(out_channels, in_channels, 3, 1, 1),

            ConvolutionalLayer(in_channels, out_channels, 1, 1, 0),
        )

    def forward(self, x):
        return self.sub_module(x)

#定义主网络
class MainNet(torch.nn.Module):

    def __init__(self):
        super(MainNet, self).__init__()

        self.trunk_52 = torch.nn.Sequential(
            ConvolutionalLayer(3, 32, 3, 1, 1),  # N 32 416 416
            DownsamplingLayer(32, 64),  # N, 64, 230, 230
            ResidualLayer(64),  #
            DownsamplingLayer(64, 128),
            ResidualLayer(128),
            ResidualLayer(128),
            DownsamplingLayer(128, 256),
            ResidualLayer(256),
            ResidualLayer(256),
            ResidualLayer(256),
            ResidualLayer(256),
            ResidualLayer(256),
            ResidualLayer(256),
            ResidualLayer(256),
            ResidualLayer(256),
        )

        self.trunk_26 = torch.nn.Sequential(
            DownsamplingLayer(256, 512),
            ResidualLayer(512),
            ResidualLayer(512),
            ResidualLayer(512),
            ResidualLayer(512),
            ResidualLayer(512),
            ResidualLayer(512),
            ResidualLayer(512),
            ResidualLayer(512),
        )

        self.trunk_13 = torch.nn.Sequential(
            DownsamplingLayer(512, 1024),
            ResidualLayer(1024),
            ResidualLayer(1024),
            ResidualLayer(1024),
            ResidualLayer(1024)
        )

        self.convset_13 = torch.nn.Sequential(
            ConvolutionalSet(1024, 512)
        )

        self.detetion_13 = torch.nn.Sequential(
            ConvolutionalLayer(512, 1024, 3, 1, 1),
            torch.nn.Conv2d(1024, 21, 1, 1, 0)    # 输出45通道
        )

        self.up_26 = torch.nn.Sequential(
            ConvolutionalLayer(512, 256, 1, 1, 0),
            # UpsampleLayer()
            Carafe(256, 32)
        )

        self.convset_26 = torch.nn.Sequential(
            ConvolutionalSet(768, 256)
        )

        self.detetion_26 = torch.nn.Sequential(
            ConvolutionalLayer(256, 512, 3, 1, 1),
            torch.nn.Conv2d(512, 21, 1, 1, 0)  # 输出45通道
        )

        self.up_52 = torch.nn.Sequential(
            ConvolutionalLayer(256, 128, 1, 1, 0),
            # UpsampleLayer()
            Carafe(128, 32)
        )

        self.convset_52 = torch.nn.Sequential(
            ConvolutionalSet(384, 128)
        )

        self.detetion_52 = torch.nn.Sequential(
            ConvolutionalLayer(128, 256, 3, 1, 1),
            torch.nn.Conv2d(256, 21, 1, 1, 0)  # 输出45通道
        )

    def forward(self, x):
        h_52 = self.trunk_52(x)  #
        h_26 = self.trunk_26(h_52)
        h_13 = self.trunk_13(h_26)
        convset_out_13 = self.convset_13(h_13)
        detetion_out_13 = self.detetion_13(convset_out_13)
        up_out_26 = self.up_26(convset_out_13)
        route_out_26 = torch.cat((up_out_26, h_26), dim=1)
        convset_out_26 = self.convset_26(route_out_26)
        detetion_out_26 = self.detetion_26(convset_out_26)
        up_out_52 = self.up_52(convset_out_26)
        route_out_52 = torch.cat((up_out_52, h_52), dim=1)
        convset_out_52 = self.convset_52(route_out_52)
        detetion_out_52 = self.detetion_52(convset_out_52)
        return detetion_out_13, detetion_out_26, detetion_out_52


if __name__ == '__main__':
    trunk = MainNet()
    print(trunk)
    # x = torch.Tensor([2,3,416,416])
    x = torch.randn([2,3,416,416],dtype=torch.float32)
    # 测试网络
    y_13, y_26, y_52 = trunk(x)
    print(y_13.shape)
    print(y_26.shape)
    print(y_52.shape)
