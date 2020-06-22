import torch
import torch.nn as nn


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

class Block35(nn.Module):
    def __init__(self,scale=1.0):
        super(Block35,self).__init__()
        self.scale = scale
        self.road0 = Conv1_1(256,32)  #17
        self.road1 = nn.Sequential(
            Conv1_1(256, 32),
            nn.Conv2d(32, 32, 3, 1,1),  # 17
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            
        )
        self.road2 = nn.Sequential(
            Conv1_1(256, 32),
            nn.Conv2d(32, 32, 3, 1,1),  # 17
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 3, 1, 1),  # 17
            nn.BatchNorm2d(32),
            nn.ReLU(True),
        )
        self.up = Conv1_1(96,256)  #17

    def forward(self,x):
        net = x
        road0 = self.road0(x)
        road1 = self.road1(x)
        road2 = self.road2(x)
        road = torch.cat((road0,road1,road2),dim=1)
        up = self.up(road)
        net = net + self.scale * up
        net = nn.ReLU(True)(net)
        return net

class Redction_a(nn.Module):   #下采样层2倍
    def __init__(self,dropout_keep_prob=0.8):
        super(Redction_a,self).__init__()
        self.road0 = nn.MaxPool2d(3,2)  #8  256
        self.road1 = nn.Sequential(
            nn.Conv2d(256, 384, 3, 2),  # 8 n
            nn.BatchNorm2d(384),
            nn.ReLU(True),
        )
        self.road2 = nn.Sequential(
            Conv1_1(256,192),
            nn.Conv2d(192, 192, 3, 1,1),  # 17
            nn.BatchNorm2d(192),
            nn.ReLU(True),
            nn.Conv2d(192, 256, 3, 2),  # 8
            nn.BatchNorm2d(256),
            nn.ReLU(True),
        )
    def forward(self,x):
        road0 =self.road0(x)
        road1 = self.road1(x)
        road2 = self.road2(x)
        net = torch.cat((road0,road1,road2),dim=1)
        return net

class Block17(nn.Module):
    def __init__(self,scale=1.0):
        super(Block17,self).__init__()
        self.scale = scale
        self.road0 = Conv1_1(896,128)
        self.road1 = nn.Sequential(
            Conv1_1(896,128),
            nn.Conv2d(128,128, 3, 1,1),  # 8 n
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.Conv2d(128, 128, 3, 1, 1),  # 8 n
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
        )
        self.up = Conv1_1(256,896)
    def forward(self,x):
        net = x
        road0 = self.road0(x)
        road1 = self.road1(x)
        road = torch.cat((road0,road1),dim=1)
        up = self.up(road)
        net = net + self.scale*up
        net = nn.ReLU(True)(net)
        return net

class Redction_b(nn.Module):   #下采样层2倍
    def __init__(self,dropout_keep_prob=0.8):
        super(Redction_b,self).__init__()
        self.road0 = nn.MaxPool2d(3,2)  # 3  896
        self.road1 = nn.Sequential(
            Conv1_1(896,256),
            nn.Conv2d(256, 384, 3, 2),  # 3 n
            nn.BatchNorm2d(384),
            nn.ReLU(True),
        )
        self.road2 = nn.Sequential(
            Conv1_1(896,256),
            nn.Conv2d(256,256,3,2),  # 3 288
            nn.BatchNorm2d(256),
            nn.ReLU(True),
        )
        self.road3 = nn.Sequential(
            Conv1_1(896,256),
            nn.Conv2d(256,288,3,1,padding=1),  # 8 288
            nn.BatchNorm2d(288),
            nn.ReLU(True),
            nn.Conv2d(288, 256, 3, 2),  # 3 n
            nn.BatchNorm2d(256),
            nn.ReLU(True),
        )

    def forward(self,x):
        road0 =self.road0(x)
        road1 = self.road1(x)
        road2 = self.road2(x)
        road3 = self.road3(x)
        net = torch.cat((road0,road1,road2,road3),dim=1)
        return net       #1792 *3*3

class Block8(nn.Module):
    def __init__(self,scale=1.0):
        super(Block8,self).__init__()
        self.scale = scale
        self.road0 = Conv1_1(1792,192)
        self.road1 = nn.Sequential(
            Conv1_1(1792,192),
            nn.Conv2d(192, 192, 3, 1,1),  # 3 n
            nn.BatchNorm2d(192),
            nn.ReLU(True),
            
        )
        self.up = Conv1_1(384,1792)

    def forward(self,x):
        net =x
        road0 = self.road0(x)
        road1 = self.road1(x)
        road = torch.cat((road0,road1),dim=1)
        up = self.up(road)
        net = net + self.scale *up
        net =nn.ReLU(True)(net)
        return net


class Inception_resnet_v1(nn.Module):
    def __init__(self):
        super(Inception_resnet_v1,self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3,32,3,2),       # 79
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32,32,3,1),         #77
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 3, 1,1),      #77
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(3,2),         #38
            Conv1_1(64,80),           #38
            nn.Conv2d(80, 192, 3, 1),  # 36
            nn.BatchNorm2d(192),
            nn.ReLU(True),
            nn.Conv2d(192, 256, 3, 2),  # 17
            nn.BatchNorm2d(256),
            nn.ReLU(True),
        )
        self.block35 = Block35(scale=0.17)
        self.reduction_a = Redction_a(dropout_keep_prob=0.8)
        self.block17 = Block17(scale=0.10)
        self.reduction_b = Redction_b(dropout_keep_prob=0.8)
        self.block8 = Block8(scale=0.20)
        self.avg_pool = nn.AvgPool2d(3,3)
        self.fc1 = nn.Linear(1792,128)
        self.fc2 = nn.Linear(128,4)  # 78为人脸分类数量
        self.center_loss = My_Center_Loss()

    def make_layer(self,block,repeat):
        layer = []
        for i in range(repeat):
            layer.append(block)
        return nn.Sequential(*layer)

    def forward(self,x):
        x = self.stem(x)
        x = self.make_layer(self.block35,5)(x)
        x = self.reduction_a(x)
        x = self.make_layer(self.block17,10)(x)
        x = self.reduction_b(x)
        x = self.make_layer(self.block8,5)(x)
        x = self.avg_pool(x)
        x = torch.flatten(x,start_dim=1,end_dim=-1)
        x = nn.Dropout(0.8)(x)
        prelogits = self.fc1(x)  # 人脸特征向量 128
        logits = self.fc2(prelogits)  # 分类数量
        # logits = torch.softmax(logits,dim=1)
        return prelogits,logits

class My_Center_Loss(nn.Module):

    def __init__(self, num_classes=4, feature_dim=128):
        super(My_Center_Loss, self).__init__()
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        self.center = nn.Parameter(torch.randn(self.num_classes, self.feature_dim), requires_grad=True)
        # self.center = torch.tensor([[1, 1], [2, 2]], dtype=torch.float32).to(self.device)  #测试是否可行

    def forward(self, input, target, lambdas):
        # input = torch.nn.functional.normalize(input)
        center_exp = self.center.index_select(dim=0, index=target.long()).cuda()
        count = torch.histc(target, bins=self.num_classes, min=0, max=self.num_classes - 1)
        count_exp = count.index_select(dim=0, index=target.long())
        loss = lambdas / 2 * torch.mean(torch.div(torch.sum(torch.pow(input - center_exp, 2), dim=1), count_exp))
        return loss


if __name__ == '__main__':
    x = torch.randn(3,3,160,160)
    net = Inception_resnet_v1()
    out = net(x)
    print(out[0].shape,out[1].shape)
    # net_35 = Block35(dropout_keep_prob=0.8)
    # out_35 = net_35(out)
    # print(out_35.shape)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    losses = My_Center_Loss().cuda()
    print("loss1",losses,losses.parameters())
    print(losses.float())
