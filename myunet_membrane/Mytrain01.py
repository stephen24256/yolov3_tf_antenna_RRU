import torch
import numpy as np
import torch.nn as nn
from MyNet01 import MyNet
from mydataset01 import Mydataset
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from PIL import Image
import torchvision.transforms as trans
# np.set_printoptions(linewidth=300,edgeitems=300,suppress=True)

class MyTrain():
    def __init__(self):
        super(MyTrain,self).__init__()
        self.net = MyNet().cuda()
        self.path = r"./membrane/train"
        self.dataset = Mydataset(self.path)

        self.img_data = DataLoader(self.dataset,batch_size=4,shuffle=True)
        self.loss = nn.BCEWithLogitsLoss()
        # self.loss = nn.BCELoss()

        self.optim = torch.optim.Adam(self.net.parameters())
        # self.optim = torch.optim.SGD(self.net.parameters(),lr=0.0001,momentum=0.99)

        self.save_modle_path = r"./model/unetv01.pth"
        self.save_img_path = r"./mytrain_img01"

    def Train(self):
        self.net.train()
        if os.path.exists(self.save_modle_path):
            self.net = torch.load(self.save_modle_path)
        else:
            print("No param")
        epoch = 100
        for j in range(epoch):
            losses = []
            for i,(img1,img2) in enumerate(self.img_data):
                img1,img2 = img1.cuda(),img2.cuda()
                output = self.net(img1)
                # print("img2",img2.cpu().numpy())
                # print("output",output.cpu().detach().numpy())
                # exit()
                loss = self.loss(output,img2)
                if i%4 ==0:
                    # losses.append(loss.float())
                    print("epoch:{0}_{1}----Loss:{2}".format(j,i,loss.item()))
                    # plt.clf()
                    # plt.plot(losses)
                    # plt.pause(0.01)
                    # plt.savefig("lossv1.jpg")
                self.optim.zero_grad()
                loss.backward()
                self.optim.step()

                if i%4 ==0:
                    # src_img = img1[0].resize((388,388))
                    target_img = img2[0]
                    output_img = output[0]
                    img = torch.stack([target_img,output_img],dim=0)
                    # print("img.shape",img.shape)
                    save_image(img.cpu(),os.path.join(self.save_img_path,"{}.png".format(i)))
                    # print("saved successfully !")
                    del img
                del img1,img2
            torch.save(self.net,self.save_modle_path)

    def te(self,path):
        net = torch.load(self.save_modle_path)
        net.eval()
        names = os.listdir(path)
        for i in range(len(names)):
            src_img_path = os.path.join(path,"{}.png".format(i))
            image = Image.open(src_img_path)
            image = image.resize((256,256))
            image = np.array(image)
            image1 = trans.ToTensor()(image)
            print(image1.shape)
            image2 = image1.unsqueeze(0).cuda()
            predcit_img = net(image2)
            predcit_img = predcit_img.squeeze(0)
            print("pr",predcit_img.shape)
            print("image1",image1.shape)
            img = torch.stack([image1,predcit_img.cpu()],dim=0)
            save_image(img.cpu(),os.path.join(path,"{}_pred.png".format(i)))

if __name__ == '__main__':
    mytrain = MyTrain()
    mytrain.Train()
    # mytrain.te(r"G:\PyCharmProjects\myunet_membrane\membrane\test")




