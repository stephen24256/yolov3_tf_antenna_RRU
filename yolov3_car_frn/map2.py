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
import utils as utils
import torch
from detector import Detector

class YoloTest(object):
    def __init__(self):
        self.input_size = 416
        self.anchor_per_scale = 3

        self.path1 = r"./data/classes/antenna.names"  #修改

        self.classes = utils.read_class_names(self.path1)
        self.num_classes = len(self.classes)

        self.score_threshold = 0.4
        self.iou_threshold = 0.5

        self.path3 = "./data/dataset/antenna_train.txt"

        self.annotation_path = self.path3

        self.write_image = True

        self.path5 = r"./data/detection1/"

        self.write_image_path = self.path5  # 是否将图片的预测结果保存
        self.show_label = True

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
                print("image_path",image_path)

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
                detector = Detector()
                path = r"G:\yolov3_05_14\data\antenna\JPEGImages"
                bboxes_pr = detector.predict(image_path)
                # exit()
                #
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
                        print("bbox",bbox)
                        # if bbox.shape[0] == 0:
                        #     continue
                        coor = np.array(bbox[:4], dtype=np.int32)
                        score = bbox[4]
                        print("bbox2", bbox)
                        class_ind = int(bbox[5])
                        class_name = self.classes[class_ind]
                        score = '%.4f' % score
                        xmin, ymin, xmax, ymax = list(map(str, coor))
                        bbox_mess = ' '.join([class_name, score, xmin, ymin, xmax, ymax]) + '\n'
                        f.write(bbox_mess)
                        print('\t' + str(bbox_mess).strip())

    def two(self,path1,path2=None):
            lines = os.listdir(path1)
            for num, line in enumerate(lines):
                annotation = line.split()
                # print("annotation109",annotation,type(annotation))
                image_path = os.path.join(path1,"".join(annotation))
                # print("image_path",image_path)
                image = cv2.imread(image_path)
                bboxes_pr = self.predict(image)
                image = utils.draw_bbox(image, bboxes_pr, show_label=self.show_label)
                if path2 !=None:
                    cv2.imwrite(path2 + "{}".format(num) + ".jpg", image)
                cv2.imshow('t', image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
if __name__ == '__main__':
    # 预测单个图像
    YoloTest().evaluate()   #预测多个图像

