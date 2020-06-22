#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : dataset.py
#   Author      : YunYang1994
#   Created date: 2019-03-15 18:05:03
#   Description :
#
#================================================================
import os
import cv2
import random
import numpy as np
import tensorflow as tf
import core.utils as utils
# from core.config import cfg
np.set_printoptions(linewidth=1000,edgeitems=500,suppress=True)


class Dataset(object):
    """implement Dataset here"""
    def __init__(self, train_or_test_datsets):
        self.path1 = r"./data/dataset/antenna_train.txt"
        self.path2 = r"./data/dataset/antenna_test.txt"
        self.annot_path  = self.path1 if train_or_test_datsets == 'train' else self.path2
        self.input_sizes = [416] if train_or_test_datsets == 'train' else 416
        self.batch_size = 4 if train_or_test_datsets == 'train' else 3
        self.data_aug = True if train_or_test_datsets == 'train' else False
        self.train_input_sizes = [416]
        self.strides = np.array([8, 16, 32])
        self.path3 = r"./data/classes/antenna.names"
        self.classes = utils.read_class_names(self.path3)
        self.num_classes = len(self.classes)
        print("num",self.num_classes)
        self.path4 = r"./data/anchors/basline_anchors.txt"
        self.anchors = np.array(utils.get_anchors(self.path4))
        # print("12self.anchor",self.anchors)
        self.anchor_per_scale = 3
        self.max_bbox_per_scale = 150

        self.annotations = self.load_annotations(train_or_test_datsets)
        self.num_samples = len(self.annotations)
        self.num_batchs = int(np.ceil(self.num_samples / self.batch_size))
        self.batch_count = 0

    # TODO 读取标注文件
    def load_annotations(self, dataset_type):
        with open(self.annot_path, 'r') as f:
            txt = f.readlines()
            annotations = [line.strip() for line in txt if len(line.strip().split()[1:]) != 0]
        np.random.shuffle(annotations)
        print("annotations",annotations,type(annotations))
        return annotations

    def __iter__(self):
        return self

    def __len__(self):
        return self.num_batchs

    # TODO 所有的输入
    def __next__(self):
        """
        每次迭代 产生 label值
        :return:
        """
        with tf.device('/GPU:0'):
            self.train_input_size = random.choice(self.train_input_sizes)       # 随机选取一种input_size

            # print("1234",self.train_input_size)
            self.train_output_sizes = self.train_input_size // self.strides

            batch_image = np.zeros((self.batch_size, self.train_input_size, self.train_input_size, 3))
            # Y3 [52,52]
            batch_label_sbbox = np.zeros((self.batch_size, self.train_output_sizes[0], self.train_output_sizes[0],
                                          self.anchor_per_scale, 5 + self.num_classes))
            # print("9self.train_output_sizes[0],",self.train_output_sizes[0])
            # Y2 [26,26]
            batch_label_mbbox = np.zeros((self.batch_size, self.train_output_sizes[1], self.train_output_sizes[1],
                                          self.anchor_per_scale, 5 + self.num_classes))
            # Y1 [13,13]
            batch_label_lbbox = np.zeros((self.batch_size, self.train_output_sizes[2], self.train_output_sizes[2],
                                          self.anchor_per_scale, 5 + self.num_classes))

            batch_sbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4))
            batch_mbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4))
            batch_lbboxes = np.zeros((self.batch_size, self.max_bbox_per_scale, 4))     #self.max_bbox_per_scale =150 ,为什么？

            num = 0
            if self.batch_count < self.num_batchs:
                while num < self.batch_size:
                    index = self.batch_count * self.batch_size + num
                    if index >= self.num_samples:
                        index -= self.num_samples
                    annotation = self.annotations[index]
                    # print("13annotation",annotation)
                    image, bboxes = self.parse_annotation(annotation)
                    # print("14image",image.shape,bboxes)
                    label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes = self.preprocess_true_boxes(bboxes) # TODO 生成数据

                    batch_image[num, :, :, :] = image
                    batch_label_sbbox[num, :, :, :, :] = label_sbbox
                    batch_label_mbbox[num, :, :, :, :] = label_mbbox
                    batch_label_lbbox[num, :, :, :, :] = label_lbbox
                    batch_sbboxes[num, :, :] = sbboxes
                    batch_mbboxes[num, :, :] = mbboxes
                    batch_lbboxes[num, :, :] = lbboxes
                    num += 1
                self.batch_count += 1
                return batch_image, batch_label_sbbox, batch_label_mbbox, batch_label_lbbox, batch_sbboxes, batch_mbboxes, batch_lbboxes
            else:
                self.batch_count = 0
                np.random.shuffle(self.annotations)
                raise StopIteration

    def random_horizontal_flip(self, image, bboxes):

        if random.random() < 0.5:
            _, w, _ = image.shape
            image = image[:, ::-1, :]    # 图像镜像
            bboxes[:, [0,2]] = w - bboxes[:, [2,0]]
            # print("11bboxes",bboxes)
        return image, bboxes

    def random_crop(self, image, bboxes):

        if random.random() < 0.5:
            h, w, _ = image.shape
            max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)

            max_l_trans = max_bbox[0]
            max_u_trans = max_bbox[1]
            max_r_trans = w - max_bbox[2]
            max_d_trans = h - max_bbox[3]

            crop_xmin = max(0, int(max_bbox[0] - random.uniform(0, max_l_trans)))
            crop_ymin = max(0, int(max_bbox[1] - random.uniform(0, max_u_trans)))
            crop_xmax = max(w, int(max_bbox[2] + random.uniform(0, max_r_trans)))
            crop_ymax = max(h, int(max_bbox[3] + random.uniform(0, max_d_trans)))

            image = image[crop_ymin : crop_ymax, crop_xmin : crop_xmax]

            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] - crop_xmin
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] - crop_ymin
            # print("121bboxes",bboxes)
        return image, bboxes

    def random_translate(self, image, bboxes):
        # print("128",image.shape)
        # cv2.imshow("image1",image)
        # cv2.waitKey(0)
        if random.random() < 0.5:
            h, w, _ = image.shape
            max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)

            max_l_trans = max_bbox[0]
            max_u_trans = max_bbox[1]
            max_r_trans = w - max_bbox[2]
            max_d_trans = h - max_bbox[3]

            tx = random.uniform(-(max_l_trans - 1), (max_r_trans - 1))
            ty = random.uniform(-(max_u_trans - 1), (max_d_trans - 1))

            M = np.array([[1, 0, tx], [0, 1, ty]])
            image = cv2.warpAffine(image, M, (w, h))

            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] + tx
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] + ty
            # print("121bboxes",bboxes)
        return image, bboxes

    def parse_annotation(self, annotation):     # 从原始各种形式的高清图片尺寸转化为网络运行的图片尺寸416*416
        line = annotation.split()
        image_path = line[0]
        if not os.path.exists(image_path):
            raise KeyError("%s does not exist ... " % image_path)
        image = np.array(cv2.imread(image_path))
        # print("15imga",image.shape) #z这里可以保存
        bboxes = np.array([list(map(int, box.split(','))) for box in line[1:]]) # 所有groud truth的框 np.array([],[],[],[])
        # print("185bboxes",bboxes.shape)
        if bboxes.shape[0]!=0 and self.data_aug==True:
            image, bboxes = self.random_horizontal_flip(np.copy(image), np.copy(bboxes))
            # print("151imga", image.shape)

            image, bboxes = self.random_crop(np.copy(image), np.copy(bboxes))
            # print("152imga", image.shape)
            image, bboxes = self.random_translate(np.copy(image), np.copy(bboxes))
            # print("153imga", image.shape)  #这里可以保存
        # print("16imga",image.shape)  # 数据增强后的尺寸
        # image_preporcess 只进行了resize，没有做scale
        image, bboxes = utils.image_preporcess(np.copy(image), [self.train_input_size, self.train_input_size], np.copy(bboxes))
        # print("17imga",image.shape)  #416*416
        # print("18bboxes",bboxes)
        # cv2.imshow("img2", image)
        # cv2.imwrite("./2.jpg", image)
        # cv2.waitKey(0)

        return image, bboxes

    def bbox_iou(self, boxes1, boxes2):

        boxes1 = np.array(boxes1)
        boxes2 = np.array(boxes2)

        boxes1_area = boxes1[..., 2] * boxes1[..., 3]
        boxes2_area = boxes2[..., 2] * boxes2[..., 3]

        boxes1 = np.concatenate([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                                boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
        boxes2 = np.concatenate([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                                boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

        left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
        right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

        inter_section = np.maximum(right_down - left_up, 0.0)
        inter_area = inter_section[..., 0] * inter_section[..., 1]
        union_area = boxes1_area + boxes2_area - inter_area

        return inter_area / union_area

    def preprocess_true_boxes(self, bboxes):
        """
        获取每个图像的 bboxes labels
        :param bboxes: 每一个图像的所有gt bboxes
        :return:
        """
        label = [np.zeros((self.train_output_sizes[i], self.train_output_sizes[i], self.anchor_per_scale,
                           5 + self.num_classes)) for i in range(3)]
        bboxes_xywh = [np.zeros((self.max_bbox_per_scale, 4)) for _ in range(3)] # [(150,4),(150,4),(150,4)] 指150行4列 三种尺度 所有的gt boxes的 x,y,w,h
        bbox_count = np.zeros((3,))
        # print("31label",label,bboxes_xywh,bbox_count)  #全部是0
        # print("32bboxes",bboxes)
        for bbox in bboxes:
            # print("b",bbox)
            bbox_coor = bbox[:4]  # GT的坐标信息
            bbox_class_ind = bbox[4]    # GT的类别

            # 如果某个Anchor Box对应这个实际的边框，那么对应的各个类别的预测概率的构建
            onehot = np.zeros(self.num_classes, dtype=np.float)
            onehot[bbox_class_ind] = 1.0
            # print("onehot",onehot)  #因为为1个类别所以全部是1
            uniform_distribution = np.full(self.num_classes, 1.0 / self.num_classes)
            # print("33uniform_distribution",uniform_distribution)   #因为为1个类别所以全部是1
            deta = 0.01
            # TODO smooth label
            smooth_onehot = onehot * (1 - deta) + deta * uniform_distribution
            # print("smooth_onehot",smooth_onehot)
            # 计算真实边框的[cx,cy,w,h]
            bbox_xywh = np.concatenate([(bbox_coor[2:] + bbox_coor[:2]) * 0.5, bbox_coor[2:] - bbox_coor[:2]], axis=-1)
            # print("34bbox_xywh",bbox_xywh)
            # 计算真实边框在Feature Map上的坐标值
            bbox_xywh_scaled = 1.0 * bbox_xywh[np.newaxis, :] / self.strides[:, np.newaxis]  # 在特征图上的xywh --> [3,4]
            # print("35bbox_xywh_scaled",bbox_xywh_scaled)
            iou = []
            exist_positive = False
            # 一个gt 都需要生成 3 个输出
            for i in range(3):# 3个分支
                # 锚框(中心点、高度和宽度)
                anchors_xywh = np.zeros((self.anchor_per_scale, 4))
                anchors_xywh[:, 0:2] = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32) + 0.5
                # print("36anchors_xywh",anchors_xywh)
                anchors_xywh[:, 2:4] = self.anchors[i]         #  self.anchors [3,3,2]
                # print("37anchors_xywh",anchors_xywh)

                # 3个数 对于3中尺度下的各iou
                iou_scale = self.bbox_iou(bbox_xywh_scaled[i][np.newaxis, :], anchors_xywh)  # 每一个gt boxes在每一种尺度下最和哪一种聚类产生的框的尺度匹配就选哪种框的wh
                # print("38",iou_scale)
                iou.append(iou_scale)
                iou_mask = iou_scale > 0.3

                # 如果有一个iou大于0.3，就执行，也可以有多个
                if np.any(iou_mask):
                    xind, yind = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32)

                    label[i][yind, xind, iou_mask, :] = 0
                    label[i][yind, xind, iou_mask, 0:4] = bbox_xywh       # 一个cell只能预测一个gt boxes
                    label[i][yind, xind, iou_mask, 4:5] = 1.0
                    label[i][yind, xind, iou_mask, 5:] = smooth_onehot

                    bbox_ind = int(bbox_count[i] % self.max_bbox_per_scale)
                    bboxes_xywh[i][bbox_ind, :4] = bbox_xywh
                    bbox_count[i] += 1

                    exist_positive = True
            # 如果没有一种anchors的尺度 于 gt的iou大于0.3，则使用最大的iou对应的xywh，确保一定有anchor与GT匹配
            if not exist_positive:
                best_anchor_ind = np.argmax(np.array(iou).reshape(-1), axis=-1)
                best_detect = int(best_anchor_ind / self.anchor_per_scale)
                best_anchor = int(best_anchor_ind % self.anchor_per_scale)
                xind, yind = np.floor(bbox_xywh_scaled[best_detect, 0:2]).astype(np.int32)

                label[best_detect][yind, xind, best_anchor, :] = 0
                label[best_detect][yind, xind, best_anchor, 0:4] = bbox_xywh
                label[best_detect][yind, xind, best_anchor, 4:5] = 1.0
                label[best_detect][yind, xind, best_anchor, 5:] = smooth_onehot

                bbox_ind = int(bbox_count[best_detect] % self.max_bbox_per_scale)
                bboxes_xywh[best_detect][bbox_ind, :4] = bbox_xywh
                bbox_count[best_detect] += 1

        label_sbbox, label_mbbox, label_lbbox = label
        sbboxes, mbboxes, lbboxes = bboxes_xywh

        # print("10",label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes)
        return label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes

if __name__ == '__main__':
    trainset = Dataset('train')
    steps_per_period = len(trainset)
    print("312",steps_per_period)
    # print("12",trainset.num_batchs)
    import torch.utils.data as data
    train_loader = next(iter(trainset))[0]
    # print("12",train_loader.shape,len(train_loader),type(train_loader))
    img1 = train_loader[0].squeeze()
    print("121",img1.shape)
    cv2.imshow("img8",img1)
    cv2.imwrite("./1.jpg",img1)
    cv2.waitKey(0)
    print("13",train_loader.shape)
    # train_datas = next(iter(trainset))
    # train_epoch_loss, test_epoch_loss = [], []
    # a = []
    # for k, x0 in enumerate(train_loader):
    #     print("k",k)
    #     print("x0",x0.shape)
    #     a.append(x0)
    # print("a",a[0],len(a),a[0].shape)
    # for j in range(7):
    #     # print("train_loader[j]",train_loader[j].shape)
    #     # print("---------")
    #     for i,x1 in enumerate(train_loader[j]):
    #         print("i",i)
    #         print(x1.shape)
            # print(x1[6],len(x1[6]))
    # for data in trainset:
    #     print("15",data[0])