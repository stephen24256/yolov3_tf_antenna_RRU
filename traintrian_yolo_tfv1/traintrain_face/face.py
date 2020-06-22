import torchvision.models as models
from torch import nn
import torch
from torch.nn import functional as F
from traintrain_face.dataset import *
from torch import optim
from torch.utils.data import DataLoader
import torch.jit as jit


class Arcsoftmax(nn.Module):
    def __init__(self, feature_num, cls_num):
        super().__init__()
        self.w = nn.Parameter(torch.randn((feature_num, cls_num)),requires_grad=True)
        self.func = nn.Softmax()

    def forward(self, x, s=1, m=0.2):
        x_norm = F.normalize(x, dim=1)
        w_norm = F.normalize(self.w, dim=0)

        cosa = torch.matmul(x_norm, w_norm) / 10
        a = torch.acos(cosa)

        arcsoftmax = torch.exp(
            s * torch.cos(a + m) * 10) / (torch.sum(torch.exp(s * cosa * 10), dim=1, keepdim=True) - torch.exp(
            s * cosa * 10) + torch.exp(s * torch.cos(a + m) * 10))

        return arcsoftmax


class FaceNet(nn.Module):

    def __init__(self):
        super(FaceNet, self).__init__()
        self.sub_net = nn.Sequential(
            models.mobilenet_v2(pretrained=True),
        )
        # print( models.mobilenet_v2())
        self.feature_net = nn.Sequential(
            nn.BatchNorm1d(1000),
            nn.LeakyReLU(0.1),
            nn.Linear(1000, 512, bias=False),
        )
        self.arc_softmax = Arcsoftmax(512, 315)

    def forward(self, x):
        y = self.sub_net(x)
        feature = self.feature_net(y)
        return feature, self.arc_softmax(feature, 1, 1)

    def encode(self, x):
        return self.feature_net(self.sub_net(x))


def compare(face1, face2):
    face1_norm = F.normalize(face1)
    face2_norm = F.normalize(face2)
    # print(face1_norm.shape)
    # print(face2_norm.shape)
    cosa = torch.matmul(face1_norm, face2_norm.T)
    return cosa

if __name__ == '__main__':

    # 训练过程
    # net = FaceNet().cuda()
    # loss_fn = nn.NLLLoss()
    # optimizer = optim.Adam(net.parameters())
    #
    # dataset = MyDataset("data")
    # dataloader = DataLoader(dataset=dataset, batch_size=128, shuffle=True)
    #
    # for epoch in range(100000):
    #     for xs, ys in dataloader:
    #         feature, cls = net(xs.cuda())
    #
    #         loss = loss_fn(torch.log(cls), ys.cuda())
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()
    #     print(torch.argmax(cls, dim=1), ys)
    #     print(str(epoch)+"Loss====>"+str(loss.item()))
    #     if epoch%100==0:
    #         torch.save(net.state_dict(), "params/2.pt")
    #         print(str(epoch)+"参数保存成功")
    #     if epoch%500==0:
    #         torch.save(net.state_dict(), "params/3.pt")
    #         print(str(epoch)+"参数保存成功")
    # 使用
    net = FaceNet()
    net.load_state_dict(torch.load("params/2.pt"))
    net.eval()
    #
    # person1 = tf(Image.open("test_img1/15.jpg")).cuda()
    # person1_feature = net.encode(torch.unsqueeze(person1,0))
    #
    # person2 = tf(Image.open("test_img1/16.jpg")).cuda()
    # person2_feature = net.encode(person2[None, ...])
    #
    # siam1 = compare(person1_feature, person2_feature)
    # print('siam',siam1)

    # person3 = tf(Image.open("test_img2/13.jpg")).cuda()
    # person3_feature = net.encode(torch.unsqueeze(person3,0))
    #
    # person4 = tf(Image.open("test_img2/142.jpg")).cuda()
    # person4_feature = net.encode(person4[None, ...])
    #
    # siam2 = compare(person3_feature, person4_feature)
    # print('siam',siam2)

    # 把模型和参数进行打包，以便C++或PYTHON调用
    x = torch.Tensor(1, 3, 112, 112)
    traced_script_module = jit.trace(net, x)
    traced_script_module.save("model.pt")

