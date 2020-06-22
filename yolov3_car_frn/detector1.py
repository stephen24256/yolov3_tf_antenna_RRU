from model import *
import cfg
import torch
import numpy as np
import PIL.Image as pimg
import PIL.ImageDraw as draw
from PIL import ImageFont
import tool
import time
# from MyNet01 import DarkNet53


class Detector(torch.nn.Module):

    def __init__(self,save_path):
        super(Detector, self).__init__()
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.net = MainNet().to(device)
        # self.net = DarkNet53(repeat=[1, 2, 8, 8, 4]).to(device)

        self.net.load_state_dict(torch.load(save_path))
        self.net.eval()

    def forward(self, input, thresh, anchors):  # 这里anchors有9个
        output_13, output_26, output_52 = self.net(input)
        # print(output_13.shape)
        # print(output_13)
        idxs_13, vecs_13 = self._filter(output_13, thresh)  # 1张图就是H W 3 15，分为H W 3 和15两个部分，以3个数值确定
        boxes_13 = self._parse(idxs_13, vecs_13, 32, anchors[13])  # 解析
        # print(boxes_13.shape)
        idxs_26, vecs_26 = self._filter(output_26, thresh)
        boxes_26 = self._parse(idxs_26, vecs_26, 16, anchors[26])
        idxs_52, vecs_52 = self._filter(output_52, thresh)
        boxes_52 = self._parse(idxs_52, vecs_52, 8, anchors[52])
        # print(torch.cat([boxes_13, boxes_26, boxes_52], dim=0).shape)
        return torch.cat([boxes_13, boxes_26, boxes_52], dim=0)

    def _filter(self, output, thresh):
        output = output.permute(0, 2, 3, 1)  # 先换轴N 45 H W --> N H W 45
        output = output.reshape(output.size(0), output.size(1), output.size(2), 3, -1)  # 再变形，N H W 45——》N H W 3 15
        mask = output[..., 0] > thresh  # 变为布尔值掩码 NHW3
        idxs = mask.nonzero()  # 取出布尔值为True的索引 # N' 4 其中N'为有多少框的置信度超过阈值， 4 为框的索引（NHW3）
        vecs = output[mask]  # 根据掩码取出对应数值 N' 15 多少个框，每个框15个值
        return idxs, vecs

    def _parse(self, idxs, vecs, t, anchors):  # 这里anchors有3个
        anchors = torch.Tensor(anchors).cuda()
        a = idxs[:, 3]  # index形状为N‘ 4， 取出框的类别：正方形，横着的，竖着的
        confidence = vecs[:, 0]  # vecs的形状为N' 15，取出每个框的置信度
        _classify = vecs[:, 5:]  # 取出每个框的中目标的类别：10类
        if len(_classify) == 0:  # 如果没有图上没有目标，类别的长度就是0 ，防止报错
            classify = torch.Tensor([])
        else:
            classify = torch.argmax(_classify, dim=1).float()  # 取出每类目标的最大值的索引，就是预测的目标类型

        cy = (idxs[:, 1].float() + vecs[:, 2]) * t  # H H'
        cx = (idxs[:, 2].float() + vecs[:, 1]) * t  # W W'
        w = anchors[a, 0] * torch.exp(vecs[:, 3])  # 取出具体的框的W，乘以W偏移率
        h = anchors[a, 1] * torch.exp(vecs[:, 4])  # 取出具体的框的H，乘以H偏移率
        x1 = cx - w / 2 # 左上角坐标
        y1 = cy - h / 2 # 左上角坐标
        x2 = x1 + w  # 右下角坐标
        y2 = y1 + h  # 右下角坐标
        x1, y1, x2, y2  = x1.cuda(), y1.cuda(), x2.cuda(), y2.cuda()
        confidence = confidence.cuda()
        classify = classify.cuda()
        out = torch.stack([confidence,x1,y1,x2,y2,classify], dim=1)  # 置信度， 坐标， 分类
        return out


if __name__ == '__main__':
    save_path = "models/net_yolo_9700.pth"
    t1 = time.time()
    # save_path = "models/net_yolo_car_frn.pth"

    detector = Detector(save_path)
    # y = detector(torch.randn(3, 3, 416, 416), 0.3, cfg.ANCHORS_GROUP)
    # print(y.shape)
    import os
    path = r"G:\yolov3_05_14\data\antenna\JPEGImages2"
    img_path = os.listdir(path)

    for p in img_path:
        img_1 = pimg.open(os.path.join(path,p))
        img_2 = img_1.convert('RGB')
        w, h = img_2.size
        larger_lenth = max(h, w)
        rate = larger_lenth / 416
        img_3 = img_2.resize((int(w / rate), int(h / rate)))
        img_4 = pimg.new("RGB", (416, 416), (0, 0, 0))
        img_4.paste(img_3)
        img_4 = np.array(img_4) / 255  # 归一化
        img_4 = torch.Tensor(img_4)  #
        img_4 = img_4.unsqueeze(0)  # 加1个维度N
        img_4 = img_4.permute(0, 3, 1, 2)  # NHWC-->NCHW
        img_4 = img_4.cuda()
        # 输入图片，置信度，3个建议框， 输出torch.stack([confidence,x1,y1,x2,y2,classify], dim=1)# 置信度， 坐标， 分类

        out_value = detector(img_4, 0.3, cfg.ANCHORS_GROUP_KMEANS)
        print(out_value.shape)
        boxes = []

        for j in range(2):
            classify_mask = (out_value[..., -1] == j)  # out_value[..., -1]形状为 N 6, 6为置信度， 坐标， 分类, 取第二维最后的数值（分类）
            # print(classify_mask.shape)
            _boxes = out_value[classify_mask]  # 取出相同类的框
            boxes.append(tool.nms(_boxes.cpu()))  # 相同类的框做nms，每类做一次，共10类 tool.nms返回torch.stack(r_boxes)，为N，5的形状
        print(len(boxes))
        for i, box in enumerate(boxes):
            cls_name_list = ["antenna","RRU"]
            try:
                img_draw = draw.ImageDraw(img_2)
                for n in range(box.shape[0]):
                    c,x1, y1, x2, y2 = box[n, 0:5] * rate  # N 5
                    print(c,x1, y1, x2, y2)
                    img_draw.rectangle((x1, y1, x2, y2), outline="red")
                    font = ImageFont.truetype(r'C:\Windows\Fonts\simsun.ttc', size=30)
                    img_draw.text((x1, y1), text=cls_name_list[i], fill='red', font=font)
            except:
                continue
        t2 = time.time()
        print("检测时间t2-t1", t2 - t1)
        img_2.save("car_frn.jpg")
        img_2.show()
