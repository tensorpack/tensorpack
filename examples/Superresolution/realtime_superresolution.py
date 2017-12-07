#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network
Wenzhe Shi, et al. https://arxiv.org/abs/1609.05158
"""

import tensorflow as tf
import cv2
import os
import argparse
from tensorpack import *
from tensorpack.utils.viz import *
from tensorpack.tfutils.summary import add_moving_summary
import tensorpack.tfutils.symbolic_functions as symbf
from tensorpack.utils.gpu import get_nr_gpu
from data_sampler import ImageDecode


BATCH = 32     # batch size
IMGSIZE = 256  # original size of images
SCALE = 4      # scale factor for super-resolution
CHANNELS = 3
HR_SIZE = IMGSIZE - (IMGSIZE % SCALE)  # valid input-size for high-res
LR_SIZE = HR_SIZE // SCALE             # input size for low-res
LR_SIZE_W = LR_SIZE_H = LR_SIZE        # hack for fully-convolutional network ...
HR_SIZE_W = HR_SIZE_H = HR_SIZE


def tf_rgb2ycbcr(rgb):
    """
    rgb in [0, 255]
    y   in [0, 255]
    cb  in [0, 255]
    cr  in [0, 255]
    """
    r, g, b = tf.unstack(rgb, 3, axis=3)

    y = r * 0.299 + g * 0.587 + b * 0.114
    cb = r * -0.1687 - g * 0.3313 + b * 0.5
    cr = r * 0.5 - g * 0.4187 - b * 0.0813

    cb += 128
    cr += 128

    ycbcr = tf.stack((y, cb, cr), axis=3)
    return ycbcr


def tf_ycbcr2rgb(ycbcr):
    """
    rgb in [0, 255]
    y   in [0, 255]
    cb  in [0, 255]
    cr  in [0, 255]
    """
    y, cb, cr = tf.unstack(ycbcr, 3, axis=3)

    cb -= 128
    cr -= 128

    r = y * 1. + cb * 0. + cr * 1.402
    g = y * 1. - cb * 0.34414 - cr * 0.71414
    b = y * 1. + cb * 1.772 + cr * 0.

    rgb = tf.stack((r, g, b), axis=3)
    return rgb


@layer_register()
def pixel_shift(x, r, color=True):
    """Do phase shifting of pixels.

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
        x = tf.concat([tf.squeeze(j, axis=1) for j in tf.split(x, h, axis=1)], 2)
        x = tf.concat([tf.squeeze(j, axis=1) for j in tf.split(x, w, axis=1)], 2)
        x = tf.reshape(x, (-1, h * r, w * r, 1), name="to_image")
        return x

    with tf.name_scope("pixel_shift"):
        if color:
            assert c == r**2 * 3, "input should have 3 * r ** 2 channels but is {}".format(c)
            xc = tf.split(x, 3, axis=3)
            channels = [helper(j, r) for j in xc]
            x = tf.concat(channels, 3)
        else:
            assert c == r**2 * 1, "input should have r ** 2 channels but is {}".format(c)
            x = helper(x, r)
        return x


class Model(ModelDesc):

    def _get_inputs(self):
        return [InputDesc(tf.float32, (None, LR_SIZE_H, LR_SIZE_W, 3), 'low'),
                InputDesc(tf.float32, (None, HR_SIZE_H, HR_SIZE_W, 3), 'high')]

    def _build_graph(self, input_vars):
        # in [0, 255]
        low, high = input_vars
        # in [0, 255]
        low_ycbcr = tf_rgb2ycbcr(low)
        high_ycbcr = tf_rgb2ycbcr(high)

        # in [-1, 1]
        low_y = tf.expand_dims(low_ycbcr[:, :, :, 0], axis=-1) / 128.0 - 1
        high_y = tf.expand_dims(high_ycbcr[:, :, :, 0], axis=-1) / 128.0 - 1

        pred_y = (LinearWrap(low_y)
                  .Conv2D('C1', 64, stride=1, kernel_shape=5, nl=tf.nn.relu)
                  .Conv2D('C2', 64, stride=1, kernel_shape=3, nl=tf.nn.relu)
                  .Conv2D('C3', 32, stride=1, kernel_shape=3, nl=tf.nn.relu)
                  .Conv2D('C4', SCALE ** 2, stride=1, kernel_shape=3, nl=tf.nn.tanh))()

        pred_y = pixel_shift("pixshift", pred_y, SCALE, color=False)
        self.cost = tf.reduce_mean(tf.squared_difference(pred_y, high_y), name='cost')

        # in [0, 255]
        scaled_ycbcr = tf.image.resize_bicubic(low_ycbcr, [HR_SIZE_H, HR_SIZE_W])
        # [-1, 1] --> [0, 255]
        pred_y = 128.0 * (pred_y + 1.)
        pred_ycbcr = tf.concat([pred_y, scaled_ycbcr[:, :, :, 1:3]], axis=3)
        pred = tf.identity(tf_ycbcr2rgb(pred_ycbcr), name='prediction')

        psnr = symbf.psnr(pred, high, 255.)

        with tf.name_scope("visualization_rgb"):
            low_up = tf.image.resize_bicubic(low, [HR_SIZE_H, HR_SIZE_W])  # bi-linear upscale lowres
            viz = (tf.concat([low_up, pred, high], 2))
            viz = tf.cast(tf.clip_by_value(viz, 0, 255), tf.uint8, name='viz')
            tf.summary.image('input,pred,gt', viz, max_outputs=max(30, BATCH))

            low_y_up = (tf.image.resize_bicubic(low_y, [HR_SIZE_H, HR_SIZE_W]) + 1.0) * 128.
            high_y_up = (high_y + 1.0) * 128.

        with tf.name_scope("visualization_y"):
            low_y_up = tf.image.grayscale_to_rgb(low_y_up)
            high_y_up = tf.image.grayscale_to_rgb(high_y_up)
            pred_y_up = tf.image.grayscale_to_rgb(pred_y)

            viz = (tf.concat([low_y_up, pred_y_up, high_y_up], 2))
            viz = tf.cast(tf.clip_by_value(viz, 0, 255), tf.uint8, name='viz2')
            tf.summary.image('input_y,pred_y,gt_y', viz, max_outputs=max(30, BATCH))

        add_moving_summary(self.cost, psnr)

    def _get_optimizer(self):
        lr = symbolic_functions.get_scalar_var('learning_rate', 0.01, summary=True)
        return tf.train.AdamOptimizer(lr)


def get_data(lmdb_path):
    ds_train = LMDBDataPoint(os.path.join(lmdb_path, 'train2017.lmdb'), shuffle=True)
    ds_train = ImageDecode(ds_train, index=0)
    augmentors = [
        imgaug.RandomCrop((HR_SIZE, HR_SIZE)),
        imgaug.Flip(horiz=True)
    ]
    ds_train = AugmentImageComponent(ds_train, augmentors, 0)
    ds_train = MapData(ds_train, lambda x: [x[0], x[0].copy()])
    ds_train = AugmentImageComponent(ds_train, [imgaug.Resize(LR_SIZE, interp=cv2.INTER_CUBIC)], 0)
    ds_train = BatchData(ds_train, BATCH)
    ds_train = PrefetchDataZMQ(ds_train, 4)

    ds_val = LMDBDataPoint(os.path.join(lmdb_path, 'val2017.lmdb'), shuffle=False)
    ds_val = FixedSizeData(ds_val, 500)
    ds_val = ImageDecode(ds_val, index=0)
    augmentors = [
        imgaug.CenterCrop((HR_SIZE, HR_SIZE))
    ]
    ds_val = AugmentImageComponent(ds_val, augmentors, 0)
    ds_val = MapData(ds_val, lambda x: [x[0], x[0].copy()])
    ds_val = AugmentImageComponent(ds_val, [imgaug.Resize(LR_SIZE, interp=cv2.INTER_CUBIC)], 0)
    ds_val = BatchData(ds_val, BATCH)
    return ds_train, ds_val


def apply(model_path, lowres_path="", highres_path="", output_path=None):
    lr = None
    if lowres_path is "":
        assert highres_path is not ""
        assert os.path.isfile(highres_path)
        hr = cv2.imread(highres_path)
        hh, ww = hr.shape[:2]
        hh = (hh // SCALE) * SCALE
        ww = (ww // SCALE) * SCALE
        print("downscale high-resolution image with shape {}x{} as input".format(hh, ww))
        hr = hr[:hh, :ww, :]
        lr = cv2.resize(hr, (0, 0), fx=.25, fy=.25, interpolation=cv2.INTER_CUBIC)
    if highres_path is "":
        assert lowres_path is not ""
        assert os.path.isfile(lowres_path)
        print("use low-resolution input")
        lr = cv2.imread(lowres_path)

    global LR_SIZE_H, LR_SIZE_W, HR_SIZE_H, HR_SIZE_W
    LR_SIZE_H, LR_SIZE_W = lr.shape[:2]
    HR_SIZE_H, HR_SIZE_W = SCALE * LR_SIZE_H, SCALE * LR_SIZE_W

    inputs = ['low']
    outputs = ['prediction']

    predict_func = OfflinePredictor(PredictConfig(
        model=Model(),
        session_init=get_model_loader(model_path),
        input_names=inputs,
        output_names=outputs))

    pred = predict_func(lr[None, ...])

    baseline = cv2.resize(lr, (0, 0), fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    if output_path:
        cv2.imwrite(output_path + "prediction.png", pred[0][0, ...])
        cv2.imwrite(output_path + "baseline.png", baseline)
    if highres_path:
        cv2.imwrite(output_path + "groundtruth.png", hr)


def get_config():
    logger.auto_set_dir()
    dataset_train, dataset_val = get_data()

    return TrainConfig(
        dataflow=dataset_train,
        callbacks=[
            ModelSaver(),
            InferenceRunner(dataset_val, [ScalarStats('cost'), ScalarStats('psnr')]),
        ],
        model=Model(),
        steps_per_epoch=dataset_train.size(),
        max_epoch=200,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use. (e.g. "--gpu 1,3")')
    parser.add_argument('--batchsize', help='load model', default=32, type=int)
    parser.add_argument('--load', help='load model')
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--lmdb_path', action='store_true')
    parser.add_argument('--lowres', help='low resolution image as input', default="", type=str)
    parser.add_argument('--highres', help='high resolution image as ground-truth', default="", type=str)
    parser.add_argument('--output', help='path for saving predicted high-res image', default="", type=str)
    args = parser.parse_args()

    BATCH = args.batchsize

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if args.apply:
        apply(args.load, args.lowres, args.highres, args.output)
    else:
        nr_tower = max(get_nr_gpu(), 1)

        config = get_config(args.lmdb_path)
        if args.load:
            config.session_init = get_model_loader(args.load)
        launch_train_with_config(config, SyncMultiGPUTrainerParameterServer(nr_tower))
