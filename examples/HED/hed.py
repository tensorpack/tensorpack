#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: hed.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import cv2
import tensorflow as tf
import argparse
import numpy as np
from six.moves import zip
import os, sys

from tensorpack import *
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.tfutils.summary import *


"""
Script to reproduce 'Holistically-Nested Edge Detection' by Saining, et al. See https://arxiv.org/abs/1504.06375.

HED is a fully-convolutional architecture. This code generally would also work
for other FCN tasks such as semantic segmentation and detection.

Usage:
    This script only needs the original BSDS dataset and applies augmentation on the fly.
    It will automatically download the dataset to $TENSORPACK_DATASET/ if not there.
    It requires pretrained vgg16 model. See the docs in `examples/load-vgg16.py`
    for instructions to convert from vgg16 caffe model.

    To view augmented images:
    ./hed.py --view

    To start training:
    ./hed.py --load vgg16.npy

    To inference (produce heatmap at each level):
    ./hed.py --load pretrained.model --run a.jpg

    To view the loss curve:
    cat train_log/hed/stat.json | jq '.[] |
    [.xentropy1,.xentropy2,.xentropy3,.xentropy4,.xentropy5,.xentropy6] |
    map(tostring) | join("\t") | .' -r | \
            ../../scripts/plot-point.py --legend 1,2,3,4,5,final --decay 0.8
"""

BATCH_SIZE = 1

class Model(ModelDesc):
    def __init__(self, is_training=True):
        self.isTrain = is_training

    def _get_input_vars(self):
        return [InputVar(tf.float32, [None, None, None] + [3], 'image'),
                InputVar(tf.int32, [None, None, None], 'edgemap') ]

    def _build_graph(self, input_vars, is_training):
        image, edgemap = input_vars
        image = image - tf.constant([104, 116, 122], dtype='float32')

        def branch(name, l, up):
            with tf.variable_scope(name) as scope:
                l = Conv2D('convfc', l, 1, kernel_shape=1, nl=tf.identity, use_bias=True)
                while up != 1:
                    l = BilinearUpSample('upsample{}'.format(up), l, 2)
                    up = up / 2
                return l

        with argscope(Conv2D, kernel_shape=3):
            l = Conv2D('conv1_1', image, 64)
            l = Conv2D('conv1_2', l, 64)
            b1 = branch('branch1', l, 1)
            l = MaxPooling('pool1', l, 2)

            l = Conv2D('conv2_1', l, 128)
            l = Conv2D('conv2_2', l, 128)
            b2 = branch('branch2', l, 2)
            l = MaxPooling('pool2', l, 2)

            l = Conv2D('conv3_1', l, 256)
            l = Conv2D('conv3_2', l, 256)
            l = Conv2D('conv3_3', l, 256)
            b3 = branch('branch3', l, 4)
            l = MaxPooling('pool3', l, 2)

            l = Conv2D('conv4_1', l, 512)
            l = Conv2D('conv4_2', l, 512)
            l = Conv2D('conv4_3', l, 512)
            b4 = branch('branch4', l, 8)
            l = MaxPooling('pool4', l, 2)

            l = Conv2D('conv5_1', l, 512)
            l = Conv2D('conv5_2', l, 512)
            l = Conv2D('conv5_3', l, 512)
            b5 = branch('branch5', l, 16)

        final_map = tf.squeeze(tf.mul(0.2, b1 + b2 + b3 + b4 + b5),
                [3], name='predmap')
        costs = []
        for idx, b in enumerate([b1, b2, b3, b4, b5, final_map]):
            output = tf.nn.sigmoid(b, name='output{}'.format(idx+1))
            xentropy = class_balanced_binary_class_cross_entropy(
                output, edgemap,
                name='xentropy{}'.format(idx+1))
            costs.append(xentropy)

        pred = tf.cast(tf.greater(output, 0.5), tf.int32, name='prediction')
        wrong = tf.cast(tf.not_equal(pred, edgemap), tf.float32)
        wrong = tf.reduce_mean(wrong, name='train_error')

        add_moving_summary(costs + [wrong])
        add_param_summary([('.*/W', ['histogram'])])   # monitor W
        self.cost = tf.add_n(costs, name='cost')

def get_data(name):
    isTrain = name == 'train'
    ds = dataset.BSDS500(name, shuffle=True)

    class CropMultiple16(imgaug.ImageAugmentor):
        def _get_augment_params(self, img):
            newh = img.shape[0] // 16 * 16
            neww = img.shape[1] // 16 * 16
            assert newh > 0 and neww > 0
            diffh = img.shape[0] - newh
            h0 = 0 if diffh == 0 else self.rng.randint(diffh)
            diffw = img.shape[1] - neww
            w0 = 0 if diffw == 0 else self.rng.randint(diffw)
            return (h0, w0, newh, neww)

        def _augment(self, img, param):
            h0, w0, newh, neww = param
            return img[h0:h0+newh,w0:w0+neww]

    if isTrain:
        shape_aug = [
            imgaug.RandomResize(xrange=(0.7,1.5), yrange=(0.7,1.5),
                aspect_ratio_thres=0.1),
            imgaug.RotationAndCropValid(90),
            CropMultiple16(),
            imgaug.Flip(horiz=True),
            imgaug.Flip(vert=True),
        ]
    else:
        # the original image shape (321x481) in BSDS is not a multiple of 16
        IMAGE_SHAPE = (320, 480)
        shape_aug = [imgaug.RandomCrop(IMAGE_SHAPE)]
    ds = AugmentImageComponents(ds, shape_aug, (0, 1))

    def f(m):
        m[m>=0.49] = 1
        m[m<0.49] = 0
        return m
    ds = MapDataComponent(ds, f, 1)

    if isTrain:
        augmentors = [
            imgaug.Brightness(63, clip=False),
            imgaug.Contrast((0.4,1.5)),
            imgaug.GaussianNoise(),
        ]
        ds = AugmentImageComponent(ds, augmentors)
    ds = BatchData(ds, BATCH_SIZE, remainder=not isTrain)
    #if isTrain:
        #ds = PrefetchDataZMQ(ds, 3)
    return ds

def view_data():
    ds = get_data('train')
    ds.reset_state()
    for ims, edgemaps in ds.get_data():
        for im, edgemap in zip(ims, edgemaps):
            assert im.shape[0] % 16 == 0 and im.shape[1] % 16 == 0, im.shape
            cv2.imshow("im", im / 255.0)
            cv2.waitKey(1000)
            cv2.imshow("edge", edgemap)
            cv2.waitKey(1000)

def get_config():
    logger.auto_set_dir()
    dataset_train = get_data('train')
    step_per_epoch = dataset_train.size() * 20
    dataset_val = get_data('val')
    #dataset_test = get_data('test')

    lr = tf.Variable(1e-5, trainable=False, name='learning_rate')
    tf.scalar_summary('learning_rate', lr)

    return TrainConfig(
        dataset=dataset_train,
        optimizer=tf.train.AdamOptimizer(lr, epsilon=1e-3),
        callbacks=Callbacks([
            StatPrinter(),
            ModelSaver(),
            HumanHyperParamSetter('learning_rate'),
            InferenceRunner(dataset_val,
                            BinaryClassificationStats('prediction',
                                                      'edgemap'))
        ]),
        model=Model(),
        step_per_epoch=step_per_epoch,
        max_epoch=500,
    )

def run(model_path, image_path):
    pred_config = PredictConfig(
            model=Model(False),
            input_data_mapping=[0],
            session_init=get_model_loader(model_path),
            output_var_names=['output' + str(k) for k in range(1, 7)])
    predict_func = get_predict_func(pred_config)
    im = cv2.imread(image_path)
    assert im is not None
    im = cv2.resize(im, (im.shape[0] // 16 * 16, im.shape[1] // 16 * 16))
    outputs = predict_func([[im.astype('float32')]])
    for k in range(6):
        pred = outputs[k][0]
        cv2.imwrite("out{}.png".format(
            '-fused' if k == 5 else str(k+1)), pred * 255)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.') # nargs='*' in multi mode
    parser.add_argument('--load', help='load model')
    parser.add_argument('--view', help='view dataset', action='store_true')
    parser.add_argument('--run', help='run model on images')
    args = parser.parse_args()

    if args.view:
        view_data()
    elif args.run:
        run(args.load, args.run)
    else:
        if args.gpu:
            os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

        config = get_config()
        if args.load:
            config.session_init = get_model_loader(args.load)
        if args.gpu:
            config.nr_tower = len(args.gpu.split(','))
        SyncMultiGPUTrainer(config).train()
