#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: load-cpm.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import cv2
import tensorflow as tf
import numpy as np
import argparse

from tensorpack import *
from tensorpack.utils.argtools import memoized

"""
15 channels:
0-1 head, neck
2-4 right shoulder, right elbow, right wrist
5-7 left shoulder, left elbow, left wrist
8-10 right hip, right knee, right ankle
11-13 left hip, left knee, left ankle
14: background
"""


def colorize(img, heatmap):
    """ img: bgr, [0,255]
        heatmap: [0,1]
    """
    heatmap = viz.intensity_to_rgb(heatmap, cmap='jet')[:, :, ::-1]
    return img * 0.5 + heatmap * 0.5


@memoized
def get_gaussian_map():
    sigma = 21
    gaussian_map = np.zeros((368, 368), dtype='float32')
    for x_p in range(368):
        for y_p in range(368):
            dist_sq = (x_p - 368 / 2) * (x_p - 368 / 2) + \
                      (y_p - 368 / 2) * (y_p - 368 / 2)
            exponent = dist_sq / 2.0 / (21**2)
            gaussian_map[y_p, x_p] = np.exp(-exponent)
    return gaussian_map.reshape((1, 368, 368, 1))


class Model(ModelDesc):
    def _get_inputs(self):
        return [InputDesc(tf.float32, (None, 368, 368, 3), 'input'),
                InputDesc(tf.float32, (None, 368, 368, 15), 'label'),
                ]

    def _build_graph(self, inputs):
        image, label = inputs
        image = image / 256.0 - 0.5

        gmap = tf.constant(get_gaussian_map())
        gmap = tf.pad(gmap, [[0, 0], [0, 1], [0, 1], [0, 0]])
        pool_center = AvgPooling('mappool', gmap, 9, stride=8, padding='VALID')
        with argscope(Conv2D, kernel_shape=3, nl=tf.nn.relu,
                      W_init=tf.random_normal_initializer(stddev=0.01)):
            shared = (LinearWrap(image)
                      .Conv2D('conv1_1', 64)
                      .Conv2D('conv1_2', 64)
                      .MaxPooling('pool1', 2)
                      # 184
                      .Conv2D('conv2_1', 128)
                      .Conv2D('conv2_2', 128)
                      .MaxPooling('pool2', 2)
                      # 92
                      .Conv2D('conv3_1', 256)
                      .Conv2D('conv3_2', 256)
                      .Conv2D('conv3_3', 256)
                      .Conv2D('conv3_4', 256)
                      .MaxPooling('pool3', 2)
                      # 46
                      .Conv2D('conv4_1', 512)
                      .Conv2D('conv4_2', 512)
                      .Conv2D('conv4_3_CPM', 256)
                      .Conv2D('conv4_4_CPM', 256)
                      .Conv2D('conv4_5_CPM', 256)
                      .Conv2D('conv4_6_CPM', 256)
                      .Conv2D('conv4_7_CPM', 128)())

        def add_stage(stage, l):
            l = tf.concat([l, shared, pool_center], 3,
                          name='concat_stage{}'.format(stage))
            for i in range(1, 6):
                l = Conv2D('Mconv{}_stage{}'.format(i, stage), l, 128)
            l = Conv2D('Mconv6_stage{}'.format(stage), l, 128, kernel_shape=1)
            l = Conv2D('Mconv7_stage{}'.format(stage),
                       l, 15, kernel_shape=1, nl=tf.identity)
            return l

        with argscope(Conv2D, kernel_shape=7, nl=tf.nn.relu):
            out1 = (LinearWrap(shared)
                    .Conv2D('conv5_1_CPM', 512, kernel_shape=1)
                    .Conv2D('conv5_2_CPM', 15, kernel_shape=1, nl=tf.identity)())
            out2 = add_stage(2, out1)
            out3 = add_stage(3, out2)
            out4 = add_stage(4, out3)
            out5 = add_stage(5, out4)
            out6 = add_stage(6, out4)
            resized_map = tf.image.resize_bilinear(out6,
                                                   [368, 368], name='resized_map')


def run_test(model_path, img_file):
    param_dict = np.load(model_path, encoding='latin1').item()
    predict_func = OfflinePredictor(PredictConfig(
        model=Model(),
        session_init=ParamRestore(param_dict),
        input_names=['input'],
        output_names=['resized_map']
    ))

    im = cv2.imread(img_file, cv2.IMREAD_COLOR).astype('float32')
    im = cv2.resize(im, (368, 368))
    out = predict_func([[im]])[0][0]
    hm = out[:, :, :14].sum(axis=2)
    viz = colorize(im, hm)
    cv2.imwrite("output.jpg", viz)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', required=True, help='.npy model file')
    parser.add_argument('--input', required=True, help='input image')
    args = parser.parse_args()
    run_test(args.load, args.input)
