#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: PatWie <mail@patwie.com>

import numpy as np
import tensorflow as tf
import os
import cv2

from tensorpack import *
from tensorpack.utils.viz import *
from tensorpack.tfutils.summary import add_moving_summary
import tensorpack.tfutils.symbolic_functions as symbf
from tensorflow.python.platform import flags
from tensorpack.models.common import layer_register

"""
Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network
"""


BATCH = 64     # batch size
IMGSIZE = 256  # original size of images
SCALE = 3      # scale factor for super-resolution
CHANNELS = 3
HR_SIZE = IMGSIZE - (IMGSIZE % SCALE)  # valid input-size for high-res
LR_SIZE = HR_SIZE // SCALE             # input size for low-res
LR_SIZE_W = LR_SIZE_H = LR_SIZE        # hack for fully-convolutional network ...
HR_SIZE_W = HR_SIZE_H = HR_SIZE

FLAGS = flags.FLAGS
tf.app.flags.DEFINE_string('gpu', "0", 'Comma separated list of GPUS that should be used. (e.g. "--gpu 1,3")')
tf.app.flags.DEFINE_string('data', "", 'path to imagenet db')
tf.app.flags.DEFINE_integer('batchsize', 32, 'images per batch')
tf.app.flags.DEFINE_string('load', "", 'load model')
tf.app.flags.DEFINE_string('lowres', "", 'low resolution image as input')
tf.app.flags.DEFINE_string('highres', None, 'high resolution image as ground-truth')
tf.app.flags.DEFINE_string('output', None, 'path for saving predicted high-res image')


def pixel_shift(x, r, color=True):
    """Do phase shifting of pixels.

    Remarks:
        TF version of pony2 in https://github.com/Tetrachrome/subpixel/blob/master/ponynet.ipynb

    Args:
        x (tf.Tensor): produced parts for final image [Batch, W, H, C * r ** 2].
        r (int): scaling factor
        color (bool, optional): produce output with 3 channels

    Returns:
        tf.tensor: final image [Batch, W*r, H*r, C]
    """
    _, h, w, c = x.get_shape().as_list()
    assert r > 0, "scaling factor has to be positive"

    def helper(x, r):
        # do the actual pixel-shift for single-channel images
        x = tf.reshape(x, (-1, h, w, r, r), name="to_strides")
        x = tf.transpose(x, (0, 1, 2, 4, 3))
        x = tf.concat_v2([tf.squeeze(j, axis=1) for j in tf.split(x, h, axis=1)], 2)
        x = tf.concat_v2([tf.squeeze(j, axis=1) for j in tf.split(x, w, axis=1)], 2)
        x = tf.reshape(x, (-1, h * r, w * r, 1), name="to_image")
        return x

    with tf.name_scope("pixel_shift"):
        if color:
            assert c == r**2 * 3, "input should have 3 * r ** 2 channels but is {}".format(c)
            xc = tf.split(x, 3, axis=3)
            channels = [helper(j, r) for j in xc]
            x = tf.concat_v2(channels, 3)
        else:
            assert c == r**2 * 1, "input should have r ** 2 channels but is {}".format(c)
            x = helper(x, r)
        return x


class Model(ModelDesc):

    def _get_input_vars(self):
        return [InputVar(tf.float32, (None, LR_SIZE_H, LR_SIZE_W, 3), 'low'),
                InputVar(tf.float32, (None, HR_SIZE_H, HR_SIZE_W, 3), 'high')]

    def _build_graph(self, input_vars):
        low, high = input_vars
        low = low / 128.0 - 1

        pred = (LinearWrap(low)
                .Conv2D('C1', 64, stride=1, kernel_shape=5, nl=tf.nn.relu)
                .Conv2D('C2', 64, stride=1, kernel_shape=3, nl=tf.nn.relu)
                .Conv2D('C3', 32, stride=1, kernel_shape=3, nl=tf.nn.relu)
                .Conv2D('C4', CHANNELS * SCALE ** 2, stride=1, kernel_shape=3, nl=tf.nn.tanh))()

        pred = pixel_shift(pred, SCALE, color=(CHANNELS == 3))
        pred = tf.multiply(pred + 1.0, 128.0, name="pred")

        psnr = symbf.psnr(pred, high, 255.)  # name is psnr/psnr
        self.cost = tf.multiply(-1.0, psnr, "cost")

        with tf.name_scope("visualization"):
            low_up = tf.image.resize_images((low + 1.0) * 128.0, [HR_SIZE_H, HR_SIZE_W])  # bi-linear upscale lowres
            viz = (tf.concat([low_up, pred, high], 2))
            viz = tf.cast(tf.clip_by_value(viz, 0, 255), tf.uint8, name='viz')
            tf.summary.image('input,pred,gt', viz, max_outputs=max(30, BATCH))

        add_moving_summary(self.cost, psnr)


def get_data():
    ds = dataset.ILSVRC12(FLAGS.data, 'train')
    ds = MapData(ds, lambda dp: [dp[0][:, :, ::-1]])
    augmentors = [
        imgaug.ResizeShortestEdge(HR_SIZE + 1),
        imgaug.RandomCrop((HR_SIZE, HR_SIZE)),
        imgaug.Flip(horiz=True)
    ]
    ds = AugmentImageComponent(ds, augmentors, 0)
    ds = MapData(ds, lambda dp: [dp[0], dp[0]])
    ds = AugmentImageComponent(ds, [imgaug.Resize(LR_SIZE, interp=cv2.INTER_CUBIC)], 0)
    ds = BatchData(ds, BATCH)
    ds = PrefetchDataZMQ(ds, 8)
    return ds


def apply(model_path, image_path, ground_truth=None, output_path=None):

    assert image_path is not ""
    lr = cv2.imread(image_path)

    # hack to use fully-convolutional network
    global LR_SIZE_H, LR_SIZE_W, HR_SIZE_H, HR_SIZE_W
    LR_SIZE_H, LR_SIZE_W = lr.shape[:2]
    HR_SIZE_H, HR_SIZE_W = 3 * LR_SIZE_H, 3 * LR_SIZE_W

    inputs = ['low']
    outputs = ['pred']
    if ground_truth:
        inputs = ['low', 'high']
        outputs = ['pred', 'psnr/psnr']

    predict_func = OfflinePredictor(PredictConfig(
        model=Model(),
        session_init=get_model_loader(model_path),
        input_names=inputs,
        output_names=outputs))

    inputs = [lr[None, ...]]
    if ground_truth:
        hr = cv2.imread(ground_truth)
        inputs = inputs + [hr[None, ...]]

    pred = predict_func(inputs)

    if output_path:
        cv2.imwrite(output_path, pred[0][0, ...])
    if ground_truth:
        print("PSNR against ground-truth is %.2fdb" % pred[1])


def get_config():
    logger.auto_set_dir()
    dataset_train = get_data()

    lr = symbf.get_scalar_var('learning_rate', 0.001, summary=True)
    return TrainConfig(
        dataflow=dataset_train,
        optimizer=tf.train.GradientDescentOptimizer(lr),
        callbacks=[ModelSaver(), StatPrinter(),
                   ScheduledHyperParamSetter('learning_rate',
                                             [(300, 1e-4), (300, 1e-5), (300, 1e-6)])],
        model=Model(),
        steps_per_epoch=2000,
        max_epoch=1000,
    )


if __name__ == '__main__':
    FLAGS = flags.FLAGS   # noqa
    FLAGS._parse_flags()

    BATCH = FLAGS.batchsize

    with change_gpu(FLAGS.gpu):
        if FLAGS.lowres is not "":
            apply(FLAGS.load, FLAGS.lowres, FLAGS.highres, FLAGS.output)
        else:
            config = get_config()
            if FLAGS.load:
                config.session_init = SaverRestore(FLAGS.load)
            SyncMultiGPUTrainer(config).train()
