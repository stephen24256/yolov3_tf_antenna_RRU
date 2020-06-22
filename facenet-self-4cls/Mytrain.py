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
        self.save_feature = []

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
                    print("feature_txt",feature_txt.shape)
                    label_text = label.cpu().numpy()
                    label_text = label_text[:,None]
                    print("label_text", label_text.shape)
                    for kk in range(feature_txt.shape[0]):
                        feature_label = np.concatenate((feature_txt[kk],label_text[kk]))
                        self.save_feature.append(feature_label)
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
            save_feature = np.array(self.save_feature)
            save_feature = torch.tensor(save_feature,dtype=torch.float32)
            print("save_feature",type(save_feature),save_feature.shape)

            a1 = []
            # path = r"./data/casia_maxpy_mtcnnpy1_182/0000102"
            path = r"./data/test/0000099_test"
            image_path = os.listdir(path)
            for image_path_i in image_path:
                image_paths = os.path.join(path,image_path_i)
                vali_feature, vali_output = self.validate(image_paths)
                # print("vali_feature",vali_feature)
                vali_feature = vali_feature.squeeze().cpu()
                # vali_feature = torch.tensor(vali_feature).type(torch.float32).cpu()
                print("vali_feature",type(vali_feature), vali_feature.shape)

                feat = []
                for n in range(save_feature.shape[0]):
                    cosa = torch.dot(vali_feature,save_feature[n][:-1]) / (torch.norm(vali_feature) * torch.norm(save_feature[n][:-1]))  # 2.余弦相似度
                    feat.append(cosa)
                print(feat)
                print(len(feat))
                print(max(feat),feat.index(max(feat)))

                index = feat.index(max(feat))
                label_per = save_feature[index][-1]
                print("label_per",label_per)
                print(save_feature[index])
                cosb = torch.dot(vali_feature, save_feature[index][:-1]) / (torch.norm(vali_feature) * torch.norm(save_feature[index][:-1]))  # 2.余弦相似度
                print("cosb",cosb)
                # exit()




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
    # train.train_v1()
    a1 = []
    # path = r"./data/casia_maxpy_mtcnnpy1_182/0000102"
    path = r"./data/test/0000045"

    image_path = os.listdir(path)
    print(image_path)
    # exit()
    for image_path_i in image_path:
        image_paths = os.path.join(path,image_path_i)
        feature, out= train.validate(image_paths)
        print("out",out)
        a1.append(out)
    print("a1",a1)