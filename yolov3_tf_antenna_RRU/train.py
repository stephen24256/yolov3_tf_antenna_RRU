#! /usr/bin/env python
# coding=utf-8
#================================================================
#   Copyright (C) 2019 * Ltd. All rights reserved.
#
#   Editor      : VIM
#   File name   : train.py
#   Author      : YunYang1994
#   Created date: 2019-02-28 17:50:26
#   Description :
#
#================================================================
import os
import time
import shutil
import numpy as np
import tensorflow as tf
import core.utils as utils
# from tqdm import tqdm
from core.dataset import Dataset
from core.yolov3 import YOLOV3
# from core.config import cfg
import matplotlib.pyplot as plt

class YoloTrain(object):
    def __init__(self):
        self.anchor_per_scale = 3

        self.path1 = r"./data/classes/antenna.names"  #修改

        self.classes = utils.read_class_names(self.path1)
        self.num_classes = len(self.classes)
        self.learn_rate_init = 1e-4     # 1e-4
        self.learn_rate_end = 1e-6     # 1e-6
        self.first_stage_epochs = 20  # 40     加载不到预训练权重，不进行一阶段训练
        self.warmup_periods = 2     # 默认为23444

        self.time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))  # 初始化时间
        self.moving_ave_decay = 0.9995    # 默认为0.9995
        self.max_bbox_per_scale = 150                           # 每个照片最多的ground truth框的数量
        self.trainset = Dataset('train')              # train
        self.testset  = Dataset('test')
        self.steps_per_period = len(self.trainset)            # 整个数据集迭代一个epoch，需要多少个batch
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

        with tf.name_scope('define_input'):
            self.input_data   = tf.placeholder(dtype=tf.float32, name='input_data')
            self.label_sbbox  = tf.placeholder(dtype=tf.float32, name='label_sbbox')
            self.label_mbbox  = tf.placeholder(dtype=tf.float32, name='label_mbbox')
            self.label_lbbox  = tf.placeholder(dtype=tf.float32, name='label_lbbox')
            self.true_sbboxes = tf.placeholder(dtype=tf.float32, name='sbboxes')
            self.true_mbboxes = tf.placeholder(dtype=tf.float32, name='mbboxes')
            self.true_lbboxes = tf.placeholder(dtype=tf.float32, name='lbboxes')
            self.trainable    = tf.placeholder(dtype=tf.bool, name='training')

        # TODO 主要的部分
        with tf.name_scope("define_loss"):
            self.model = YOLOV3(self.input_data, self.trainable)
            self.net_var = tf.global_variables()
            self.giou_loss, self.conf_loss, self.prob_loss = self.model.compute_loss(
                                                    self.label_sbbox,  self.label_mbbox,  self.label_lbbox,
                                                    self.true_sbboxes, self.true_mbboxes, self.true_lbboxes)
            self.loss = self.giou_loss + self.conf_loss + self.prob_loss

        # warm up
        with tf.name_scope('learn_rate'):
            self.global_step = tf.Variable(1.0, dtype=tf.float32, trainable=False, name='global_step')
            # 指数衰减lr
            self.learn_rate = tf.train.exponential_decay(learning_rate=self.learn_rate_init,
                      global_step=self.global_step,
                      decay_steps=self.steps_per_period * 20,
                      decay_rate=0.9,
                      staircase=False,
                      name=None)
            global_step_update = tf.assign_add(self.global_step, 1.0)

        with tf.name_scope("define_weight_decay"):
            moving_ave = tf.train.ExponentialMovingAverage(self.moving_ave_decay).apply(tf.trainable_variables())

        with tf.name_scope("define_second_stage_train"):
            second_stage_trainable_var_list = tf.trainable_variables()
            second_stage_optimizer = tf.train.AdamOptimizer(self.learn_rate).minimize(self.loss,
                                                      var_list=second_stage_trainable_var_list)

            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                with tf.control_dependencies([second_stage_optimizer, global_step_update]):
                    with tf.control_dependencies([moving_ave]):
                        self.train_op_with_all_variables = tf.no_op()      #  tf.no_op()表示执行完control_dependencies的变量更新之后，不做任何操作，主要确保control_dependencies的变量更新。

        with tf.name_scope('loader_and_saver'):
            self.loader = tf.train.Saver(self.net_var)
            self.saver  = tf.train.Saver(tf.global_variables(), max_to_keep=5)

    def train(self):
        # a、模型恢复 或者变量初始化
        checkpoint_dir = "./checkpoint/model"
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            print('加载磁盘中持久化的模型，继续训练!')
        else:
            self.sess.run(tf.global_variables_initializer())
            print('没有持久化模型，随机初始化,开始训练!')

        for epoch in range(1, 1+self.first_stage_epochs):
            train_op = self.train_op_with_all_variables
            train_epoch_loss, test_epoch_loss = [], []

            for train_data in self.trainset:
                print("21train_data[0]",train_data[0].shape)
                _, train_step_loss = self.sess.run(
                    [train_op, self.loss,],feed_dict={
                                                self.input_data:   train_data[0],
                                                self.label_sbbox:  train_data[1],
                                                self.label_mbbox:  train_data[2],
                                                self.label_lbbox:  train_data[3],
                                                self.true_sbboxes: train_data[4],
                                                self.true_mbboxes: train_data[5],
                                                self.true_lbboxes: train_data[6],
                                                self.trainable:    True,
                })
                train_epoch_loss.append(train_step_loss)
                print("{},Loss:{}".format(epoch,train_epoch_loss))
                # plt.clf()
                # plt.plot(train_epoch_loss)
                # plt.pause(0.01)
                # plt.savefig("loss1")

            print("完成训练第{}次".format(epoch))
            for test_data in self.testset:
                test_step_loss = self.sess.run( self.loss, feed_dict={
                                                self.input_data:   test_data[0],
                                                self.label_sbbox:  test_data[1],
                                                self.label_mbbox:  test_data[2],
                                                self.label_lbbox:  test_data[3],
                                                self.true_sbboxes: test_data[4],
                                                self.true_mbboxes: test_data[5],
                                                self.true_lbboxes: test_data[6],
                                                self.trainable:    False,
                })

                test_epoch_loss.append(test_step_loss)
                print("test loss: %.2f" % test_step_loss)

            print("完成测试第{}次".format(epoch))

            train_epoch_loss, test_epoch_loss = np.mean(train_epoch_loss), np.mean(test_epoch_loss)
            ckpt_file = "./checkpoint/model/yolov3_test_loss=%.4f.ckpt" % test_epoch_loss
            log_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
            print("=> Epoch: %2d Time: %s Train loss: %.2f Test loss: %.2f Saving %s ..."
                            %(epoch, log_time, train_epoch_loss, test_epoch_loss, ckpt_file))
            self.saver.save(self.sess, ckpt_file, global_step=(epoch+980))


if __name__ == '__main__':
    YoloTrain().train()