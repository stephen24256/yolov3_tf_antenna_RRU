import torch
import numpy as np
import torchvision.transforms as trans
from PIL import Image
import os
from torch.utils.data import DataLoader,Dataset
from torchvision.utils import save_image

class Mydataset(Dataset):
    def __init__(self,path):
        self.path = path
        self.names = os.listdir(os.path.join(self.path,'image'))
        self.input_size = 572   # 输入网络的图片大小
        self.label_size = 388    # 网络输出的图片大小
        self.trans = trans.Compose([
            trans.ToTensor(),
        ])

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        name = self.names[index]
        # print("name",name)

        input_img_path = os.path.join(self.path,'image')
        label_img_path = os.path.join(self.path,'label')
        input_img = Image.open(os.path.join(input_img_path,name))
        label_img = Image.open(os.path.join(label_img_path,name))
        input_img = input_img.resize((self.input_size,self.input_size))
        label_img = label_img.resize((self.label_size,self.label_size))
        input_img = self.trans(input_img)
        label_img = self.trans(label_img)

        return input_img,label_img


if __name__ == '__main__':
    path = r"G:\PyCharmProjects\myunet_membrane\membrane\train"
    mydataset = Mydataset(path)
    print(len(mydataset))
    # i = 0
    # for img1,img2 in mydataset:
    #     save_image(img1,"./img/{}.jpg".format(i),nrow=1)
    #     save_image(img2,"./img/{}.png".format(i),nrow=1)
    #     i +=1

    data = DataLoader(mydataset,batch_size=1,shuffle=True)
    for img1,img2 in data:
        print("img1",img1,img1.shape)
        img1 = img1.squeeze()
        img1 = trans.ToPILImage()(img1)
        img1.show()
        print("img2",img2,img2.shape)
        img2 = img2.squeeze()
        img2 = trans.ToPILImage()(img2)
        img2.show()
        # img2 = np.array(img2)
        # img2 = torch.tensor(img2)
        # print("2",img2.shape)
        # # b = torch.where(img2 < 0.5, torch.ones_like(img2), torch.zeros_like(img2))
        # print("mask",b)
        # b = b[None,:,:]
        # print(b.shape,type(b))
        # img3 = trans.ToPILImage()(b)
        # img3.show()
        # exit()

