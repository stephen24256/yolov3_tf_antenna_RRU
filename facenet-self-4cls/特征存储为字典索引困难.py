import numpy as np
import torch
from MyNet01 import Inception_resnet_v1
from torch.utils.data import DataLoader
import os
from Mydataset import Mydataset
import torch.nn as nn
# from centerloss import Center_Loss
from PIL import Image
from torchvision.transforms import transforms

class Train():
    def __init__(self):
        self.model_path = "./param/incep_v4_nodropout.pth"
        self.data_path = "./data/casia_maxpy_mtcnnpy1_182"
        self.train_dataset =Mydataset(self.data_path,data_aug=True)
        self.batch_size = 50
        self.epoch = 1
        self.save_feature = {}

        self.train_data = DataLoader(self.train_dataset,batch_size=self.batch_size,shuffle=True,num_workers=4)
        self.net = Inception_resnet_v1().cuda()
        self.soft_loss = nn.CrossEntropyLoss()
        # self.soft_loss = nn.NLLLoss()
        self.optim_soft = torch.optim.Adam(self.net.parameters())
        self.optim_center = torch.optim.SGD(self.net.center_loss.parameters(),lr=0.2)
    def train_v1(self):
        if os.path.exists(self.model_path):
            self.net.load_state_dict(torch.load(self.model_path))
        else:
            print("NO pararm!")
        with open("feature1.txt","w") as f:
            for i in range(self.epoch):
                for j,(image,label) in enumerate(self.train_data):
                    image,label = image.cuda(),label.cuda()
                    feature, output = self.net(image)
                    feature_txt = feature.cpu().detach().numpy()
                    label_text = label.cpu().numpy()
                    for kk,feature_per in enumerate(feature_txt):
                        self.save_feature[str(feature_per)]=label_text[kk]
                        # print("保存的特征向量", self.save_feature)

                    print("feature_txt",feature_txt.shape)
                    for txet in feature_txt:
                        f.write("{0}:{1}\n".format(txet,label[j]))
                    soft_loss = self.soft_loss(output, label)
                    label = label.float()
                    cen_loss = self.net.center_loss(feature,label,1.5)
                    loss = soft_loss + cen_loss

                    self.optim_soft.zero_grad()
                    self.optim_center.zero_grad()
                    loss.backward()
                    # soft_loss.backward(create_graph=True)
                    # cen_loss.backward()
                    self.optim_soft.step()
                    self.optim_center.step()
                    if i % 20 == 0:
                        print("epoch:", i, "j:", j, "total:", loss.item(), "softmax_loss:", soft_loss.item(),
                              "center_loss:", cen_loss.item())
                        torch.save((self.net.state_dict()), self.model_path)
            # print("save_feature",save_feature)
            print("保存的特征向量",len(self.save_feature))
            a1 = []
            path = r"./data/casia_maxpy_mtcnnpy1_182/0000102"
            image_path = os.listdir(path)
            for image_path_i in image_path:
                image_paths = os.path.join(path,image_path_i)
                vali_feature,vali_output =self.validate(image_paths)
                # print("vali_feature",vali_feature)
                vali_feature = vali_feature.squeeze()
                vali_feature = torch.tensor(vali_feature).type(torch.float32).cpu()
                print("vali_feature",vali_feature.shape)
                # exit()
            # #     a1.append(vali_feature)
            # # a1 = np.array(a1)
            # # print("a1",a1.shape)
                cosa = []
                for key, vule in self.save_feature.items():
                    print("vule",vule)

                    # print("key", key,type(key))
                    key = key[1:len(key)-1]
                    key = key.split()
                    # print("key1",key)
                    key_feature = []
                    for key1 in key:
                        key_feature.append(float(key1))
                    key_feature = np.array(key_feature)
                    key_feature = torch.tensor(key_feature).type(torch.float32)
                    print("key_feature",key_feature.shape)
                    # print(self.save_feature[str(key_feature)])
                    a = torch.dot(vali_feature,key_feature)/(torch.norm(vali_feature)*torch.norm(key_feature))     # 2.余弦相似度
                    print("a",a)
                    cosa.append(a)
                    # exit()
                print("cosa",cosa,len(cosa))
                print(max(cosa),cosa.index(max(cosa)))

                exit()




            # return self.save_feature

    def validate(self,image_path):
        with torch.no_grad() as grad:
            self.net.load_state_dict(torch.load(self.model_path))
            self.net.eval()
            image = Image.open(image_path)
            image = image.resize((160,160),Image.ANTIALIAS)
            image = transforms.ToTensor()(image)
            image = image.unsqueeze(dim=0)
            image = image.cuda()
            feature, output = self.net(image)
            # print(output)
            output = torch.softmax(output,dim=1)
            # print(output)
            out = torch.argmax(output,dim=1)
            # print(out)

            return feature,out

if __name__ == '__main__':
    train = Train()
    train.train_v1()
    # a1 = []
    # path = r"./data/casia_maxpy_mtcnnpy1_182/0000102"
    # image_path = os.listdir(path)
    # print(image_path)
    # # exit()
    # for image_path_i in image_path:
    #     image_paths = os.path.join(path,image_path_i)
    #     a2 = train.validate(image_paths)
    #     print("a2",a2)
    #     a1.append(a2)
    # print("a1",a1)