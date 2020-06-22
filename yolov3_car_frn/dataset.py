import torch
from torch.utils.data import Dataset,DataLoader
import torchvision
import numpy as np
import cfg
import os
from PIL import Image
import math

LABEL_FILE_PATH = "data/antenna_label.txt"
IMG_BASE_DIR = "data/images"

transforms = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])


def one_hot(cls_num, i):
    b = np.zeros(cls_num)
    b[i] = 1.
    return b


class MyDataset(Dataset):

    def __init__(self):
        with open(LABEL_FILE_PATH) as f:
            self.dataset = f.readlines()  # 所有行都读到

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        # 反算回去的建议框只有9种并且是固定大小的，预测框是这9种框中心点和宽高偏移后得到的，与真实框做损失
        labels = {}
        line = self.dataset[index]  # 拿到其中一行的数据
        strs = line.split()
        _img_data = Image.open(os.path.join(IMG_BASE_DIR, strs[0]))
        img_data = transforms(_img_data)
        # _boxes = np.array(float(x) for x in strs[1:])
        _boxes = np.array(list(map(float, strs[1:])))  # 转变数据类型
        boxes = np.split(_boxes, len(_boxes) // 5)  # 切分成N组，得到N个2维列表

        for feature_size, anchors in cfg.ANCHORS_GROUP_KMEANS.items():  # 取出每种特征图尺寸及对应的三种建议框
            # 生成标签：每个尺寸的标签都要生成，形状为：w,h,3,5+10
            labels[feature_size] = np.zeros(shape=(feature_size, feature_size, 3, 5 + cfg.CLASS_NUM))
            for box in boxes:  # 原图上的框，每张图片有多个
                cls, cx, cy, w, h = box  # 分类， 中心点， 宽，高
                # 除法，小数、整数分开。原图上的一个框生成大中小的三中框，每种框三个。
                # cx_index:中心点落在哪个格子内，格子左上角坐标除以特征图对应的步长。cx_offset：中心点相对于格子左上角的偏移率
                cx_offset, cx_index = math.modf(cx * feature_size / cfg.IMG_WIDTH)
                # cfg.IMG_WIDTH / feature_size实际上就是步长
                cy_offset, cy_index = math.modf(cy * feature_size / cfg.IMG_WIDTH)
                for i, anchor in enumerate(anchors):  # 取出正方形，横，竖三种框
                    anchor_area = cfg.ANCHORS_GROUP_AREA[feature_size][i]  # 每种特征图对应的正方形、横、竖三种建议框的面积
                    p_w, p_h = w / anchor[0], h / anchor[1]  # 原图框高和宽相对于建议框的偏移率
                    p_area = w * h  # 原图框的面积
                    iou = min(p_area, anchor_area) / max(p_area, anchor_area)  # 小框比大框
                    # cy_index对应高， cx_index对应宽
                    labels[feature_size][int(cy_index), int(cx_index), i] = np.array(
                        [iou, cx_offset, cy_offset, np.log(p_w), np.log(p_h), *one_hot(cfg.CLASS_NUM, int(cls))])#10,i
        return labels[13], labels[26], labels[52], img_data


if __name__ == '__main__':
    x=one_hot(10,2)
    # print(x)
    # print(*x)
    data = MyDataset()
    dataloader = DataLoader(data, 2, shuffle=True)
    # for i,x in enumerate(dataloader):
        # print(x[0].shape)
        # print(x[1].shape)
        # print(x[2].shape)
        # print(x[3].shape)
        # print(i)
        # print(x[0][...,0].shape)
        # print(x[0][0, 0, 0, 0, 1])
    for target_13, target_26, target_52, img_data in dataloader:
        print(target_13)
        print(target_26.shape)
        print(target_52.shape)
        print(img_data.shape)
