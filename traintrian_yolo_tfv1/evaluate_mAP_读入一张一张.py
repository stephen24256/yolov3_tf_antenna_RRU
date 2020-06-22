#! /usr/bin/env python
# coding=utf-8
# ================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : evaluate_mAP.py
#   Author      : YunYang1994
#   Created date: 2019-02-21 15:30:26
#   Description :
#
# ================================================================

import cv2
import os
import shutil
import numpy as np
import tensorflow as tf
import core.utils as utils
# from core.config import cfg
from core.yolov3 import YOLOV3
import torch
from PIL import Image


class YoloTest(object):
    def __init__(self):
        self.input_size = 416
        self.anchor_per_scale = 3

        self.path1 = r"./data/classes/antenna.names"  #修改

        self.classes = utils.read_class_names(self.path1)
        self.num_classes = len(self.classes)

        self.path2 = r"./data/anchors/basline_anchors.txt"

        self.anchors = np.array(utils.get_anchors(self.path2))
        self.score_threshold = 0.4
        self.iou_threshold = 0.5
        self.moving_ave_decay = 0.9995

        self.path3 = "./data/dataset/antenna_train.txt"

        self.annotation_path = self.path3

        # self.path4 = r'checkpoint1/yolov3_test_loss=2.6758.ckpt-1435'
        self.path4 = r'checkpoint1/yolov3_test_loss=1.1578.ckpt-1471'


        self.weight_file = self.path4
        self.write_image = True

        self.path5 = r"./data/detection1/"

        self.write_image_path = self.path5  # 是否将图片的预测结果保存
        self.show_label = True
        with tf.name_scope('input'):
            self.input_data = tf.placeholder(dtype=tf.float32, name='input_data')
            self.trainable = tf.placeholder(dtype=tf.bool, name='trainable')
        print("in",self.input_data.shape)
        model = YOLOV3(self.input_data, self.trainable)
        self.pred_sbbox, self.pred_mbbox, self.pred_lbbox = model.pred_sbbox, model.pred_mbbox, model.pred_lbbox
        print(1+2)
        with tf.name_scope('ema'):
            ema_obj = tf.train.ExponentialMovingAverage(self.moving_ave_decay)

        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
        self.saver = tf.train.Saver(ema_obj.variables_to_restore())
        self.saver.restore(self.sess, self.weight_file)

    def predict(self, image):

        org_image = np.copy(image)
        org_h, org_w, _ = org_image.shape   #

        image_data = utils.image_preporcess(image, [self.input_size, self.input_size])
        image_data = image_data[np.newaxis, ...]
        pred_sbbox, pred_mbbox, pred_lbbox = self.sess.run(
            [self.pred_sbbox, self.pred_mbbox, self.pred_lbbox],
            feed_dict={
                self.input_data: image_data,
                self.trainable: False
            }
        )
        pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + self.num_classes)),
                                    np.reshape(pred_mbbox, (-1, 5 + self.num_classes)),
                                    np.reshape(pred_lbbox, (-1, 5 + self.num_classes))], axis=0)
        bboxes = utils.postprocess_boxes(pred_bbox, (org_h, org_w), self.input_size, self.score_threshold)
        bboxes = utils.nms(bboxes, self.iou_threshold)

        return bboxes

    def evaluate(self):
        predicted_dir_path = './mAP/predicted'
        ground_truth_dir_path = './mAP/ground-truth'
        if os.path.exists(predicted_dir_path): shutil.rmtree(predicted_dir_path)
        if os.path.exists(ground_truth_dir_path): shutil.rmtree(ground_truth_dir_path)
        if os.path.exists(self.write_image_path): shutil.rmtree(self.write_image_path)
        os.mkdir(predicted_dir_path)
        os.mkdir(ground_truth_dir_path)
        os.mkdir(self.write_image_path)

        with open(self.annotation_path, 'r') as annotation_file:
            for num, line in enumerate(annotation_file):
                annotation = line.strip().split()
                # print("annotation109",annotation)
                image_path = annotation[0]
                image_name = image_path.split('/')[-1]
                image = cv2.imread(image_path)
                bbox_data_gt = np.array([list(map(int, box.split(','))) for box in annotation[1:]])

                if len(bbox_data_gt) == 0:
                    bboxes_gt = []
                    classes_gt = []
                else:
                    bboxes_gt, classes_gt = bbox_data_gt[:, :4], bbox_data_gt[:, 4]
                ground_truth_path = os.path.join(ground_truth_dir_path, str(num) + '.txt')

                print('=> ground truth of %s:' % image_name)
                num_bbox_gt = len(bboxes_gt)
                with open(ground_truth_path, 'w') as f:
                    for i in range(num_bbox_gt):
                        class_name = self.classes[classes_gt[i]]
                        xmin, ymin, xmax, ymax = list(map(str, bboxes_gt[i]))
                        bbox_mess = ' '.join([class_name, xmin, ymin, xmax, ymax]) + '\n'
                        f.write(bbox_mess)
                        print('\t' + str(bbox_mess).strip())
                print('=> predict result of %s:' % image_name)
                predict_result_path = os.path.join(predicted_dir_path, str(num) + '.txt')
                # print("image",image)
                bboxes_pr = self.predict(image)

                if self.write_image:
                    image = utils.draw_bbox(image, bboxes_pr, show_label=self.show_label)
                    cv2.imwrite(self.write_image_path + "{}".format(num)+".jpg", image)
                    # cv2.imshow('t', image)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()
                    # exit()
                print("bboxes_pr",bboxes_pr,len(bboxes_pr))
                with open(predict_result_path, 'w') as f:
                    for bbox in bboxes_pr:
                        print("bbox",bbox,bbox.shape)
                        coor = np.array(bbox[:4], dtype=np.int32)
                        score = bbox[4]
                        class_ind = int(bbox[5])
                        class_name = self.classes[class_ind]
                        score = '%.4f' % score
                        xmin, ymin, xmax, ymax = list(map(str, coor))
                        bbox_mess = ' '.join([class_name, score, xmin, ymin, xmax, ymax]) + '\n'
                        f.write(bbox_mess)
                        print('\t' + str(bbox_mess).strip())

    def two(self,path1,path2=None):
        lines = os.listdir(path1)
        lines.sort(key=lambda x: int(x[:-4]))
        total_per_name = []
        total_index = []
        total_cls = {}
        for num, line in enumerate(lines):
            annotation = line.split()
            image_name = str(annotation).split('_')
            image_names = image_name[0][2:]
            index = image_name[1].split('.')[0]
            total_index.append(int(index))
            total_per_name.append(image_names)
        per_name = list(set(total_per_name))
        per_name.sort(key=total_per_name.index)
        start_ind = 0
        for i, per_image_name in enumerate(per_name):
            end_ind = total_per_name.count(per_name[i]) + start_ind
            total_cls[per_image_name] = total_index[start_ind:end_ind]
            start_ind += end_ind
        for i,(key,value) in enumerate(total_cls.items()):
            total_bboxes_pr = []
            for k in value:
                image_path = os.path.join(path1,key+"_"+"{}".format(k)+".jpg")
                image = cv2.imread(image_path)
                bboxes_pr = self.predict(image)
                x_index = int(k)
                if x_index ==0:
                    for boxes in bboxes_pr:
                        x1, y1, x2, y2, cls, _ = boxes
                        origial_x1 = x1
                        origial_y1 = y1 + 552
                        origial_x2 = x2
                        origial_y2 = y2 +552
                        total_bboxes_pr.append(np.array([origial_x1, origial_y1, origial_x2, origial_y2, cls, _]))
                else:
                    for boxes in bboxes_pr:
                        x1, y1, x2, y2, cls, _ = boxes
                        origial_x1 = x1 + (832 - 60)*x_index
                        origial_y1 = y1 + 552
                        origial_x2 = x2 + (832 - 60)*x_index
                        origial_y2 = y2 +552
                        total_bboxes_pr.append(np.array([origial_x1, origial_y1, origial_x2, origial_y2, cls, _]))
            origial_bboxes = []
            for pr_bboxes in total_bboxes_pr:
                bboxes1 = pr_bboxes.tolist()
                origial_bboxes.append(bboxes1)
            origial_bboxes = torch.tensor(origial_bboxes)
            best_bboxes = utils.py_nms(origial_bboxes, 0.1)
            print("best_bboxes",len(best_bboxes))
            best_bboxes = best_bboxes.numpy().tolist()
            origial_path = r'./docs/trian_data_0610/'
            origial_path_path = os.path.join(origial_path,key+".bmp")
            origial_image = cv2.imread(origial_path_path)
            big_origial_image = utils.draw_bbox(origial_image, best_bboxes, show_label=self.show_label)
            if path2 !=None:
                save_image_path = os.path.join(path2,key+'.bmp')
                cv2.imwrite(save_image_path, big_origial_image)
                print("保存成功")


if __name__ == '__main__':
    # 预测单个图像
    # path = r'./docs/images/{}.jpg'.format(0)
    # # path = r"G:\yolov3_05_14\data\antenna\JPEGImages\2.jpg"
    # image = cv2.imread(path)
    # # image = np.array(image)
    # # print("13",image)
    # bboxes_pr = YoloTest().predict(image)
    # print("bboxes_pr",bboxes_pr)
    # # 绘图
    # image = utils.draw_bbox(image, bboxes_pr, show_label=False)
    # cv2.imshow('t', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    path = r'./docs/corp_img2/'
    # # path1 = r"docs\antenna"
    path2 = r'./docs/corp_img2pre/'
    #
    YoloTest().two(path,path2)

    # path3 = r'./docs/img/'
    # YoloTest().concat_img(path3)


    # # # 效果评估
    # YoloTest().evaluate()   #预测多个图像

