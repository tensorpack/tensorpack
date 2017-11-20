#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Patrick Wieschollek <mail@patwie.com>

import os
import cv2
import argparse
import numpy as np
from tensorpack import *
from tensorpack.tfutils.scope_utils import under_variable_scope, auto_reuse_variable_scope
import tensorflow as tf

"""
Let there be Color!: Joint End-to-end Learning of Global and Local Image Priors
for Automatic Image Colorization with Simultaneous Classification
<http://hi.cs.waseda.ac.jp/~iizuka/projects/colorization/en/>
"""

BATCH_SIZE = 32
ALPHA = 1. / 300


@layer_register()
def FusionLayer(mid_level_features, global_features):
    _, H, W, C = mid_level_features.get_shape().as_list()
    _, D = global_features.get_shape().as_list()

    global_features = tf.expand_dims(global_features, axis=1)
    global_features = tf.expand_dims(global_features, axis=1)

    tiled_glob_features = tf.tile(global_features, [1, H, W, 1])
    fused = tf.concat([mid_level_features, tiled_glob_features],
                      axis=3, name='fused_features')
    fused = Conv2D('fuse_matrix', fused, C,
                   kernel_shape=1, stride=1, nl=tf.nn.relu)
    return fused


def rgb2lab(rgb):
    """
    rgb in [0, 255]
    L   in [0, 100]
    A   in [-86.185, 98,254]  ~ [-100, 100]
    B   in [-107.863, 94.482] ~ [-100, 100]
    """
    rgb = rgb / 255.
    r, g, b = tf.unstack(rgb, 3, axis=3)

    def scale0(x):
        # r = (r > 0.04045) ? Math.pow((r + 0.055) / 1.055, 2.4) : r / 12.92;
        return tf.where(x > 0.04045, ((x + 0.055) / 1.055) ** 2.4, x / 12.92)

    r = scale0(r)
    g = scale0(g)
    b = scale0(b)

    x = (r * 0.4124 + g * 0.3576 + b * 0.1805) / 0.95047
    y = (r * 0.2126 + g * 0.7152 + b * 0.0722) / 1.00000
    z = (r * 0.0193 + g * 0.1192 + b * 0.9505) / 1.08883

    def scale1(x):
        return tf.where(x > 0.008856, x**(1. / 3.), (7.787 * x) + 16. / 116.)

    x = scale1(x)
    y = scale1(y)
    z = scale1(z)

    l = (116 * y) - 16
    a = 500 * (x - y)
    b = 200 * (y - z)

    lab = tf.stack((l, a, b), axis=3)

    return lab


def lab2rgb(lab):
    l, a, b = tf.unstack(lab, 3, axis=3)

    y = (l + 16.) / 116.
    x = a / 500. + y
    z = y - b / 200.

    def scale0(x):
        return tf.where(x * x * x > 0.008856, x * x * x, (x - 16. / 116.) / 7.787)

    x = 0.95047 * scale0(x)
    y = 1.00000 * scale0(y)
    z = 1.08883 * scale0(z)

    r = x * 3.2406 + y * -1.5372 + z * -0.4986
    g = x * -0.9689 + y * 1.8758 + z * 0.0415
    b = x * 0.0557 + y * -0.2040 + z * 1.0570

    def scale1(x):
        return tf.where(x > 0.0031308, (1.055 * (x ** (1 / 2.4)) - 0.055), 12.92 * x)

    r = tf.clip_by_value(scale1(r), 0., 1.)
    g = tf.clip_by_value(scale1(g), 0., 1.)
    b = tf.clip_by_value(scale1(b), 0., 1.)

    rgb = tf.stack((r, g, b), axis=3) * 255
    return rgb


SHAPE = 224


class Model(ModelDesc):

    def __init__(self, H=224, W=224):
        super(Model, self).__init__()
        self.HSHAPE = H
        self.WSHAPE = W

    def _get_inputs(self):
        return [InputDesc(tf.float32, (None, self.HSHAPE, self.WSHAPE, 3), 'rgb'),
                InputDesc(tf.int32, (None, ), 'labels')]

    def _build_graph(self, inputs):
        ctx = get_current_tower_context()
        given_rgb, labels = inputs
        labels = tf.squeeze(labels)

        given_lab = rgb2lab(given_rgb)
        given_l = tf.expand_dims(given_lab[:, :, :, 0], axis=-1)
        given_ab = given_lab[:, :, :, 1:]

        @auto_reuse_variable_scope
        @under_variable_scope()
        def low_level(x):
            with argscope(Conv2D, kernel_shape=3, nl=BNReLU):
                x = Conv2D('conv1', x, 64, stride=2)
                x = Conv2D('conv2', x, 128, stride=1)
                x = Conv2D('conv3', x, 128, stride=2)
                x = Conv2D('conv4', x, 256, stride=1)
                x = Conv2D('conv5', x, 256, stride=2)
                x = Conv2D('conv6', x, 512, stride=1)
            return x

        @under_variable_scope()
        def mid_level(x):
            with argscope(Conv2D, kernel_shape=3, nl=BNReLU):
                x = Conv2D('conv1', x, 512, stride=1)
                x = Conv2D('conv2', x, 256, stride=1)
            return x

        @under_variable_scope()
        def global_features(x):
            with argscope(Conv2D, kernel_shape=3, nl=BNReLU):
                x = Conv2D('conv1', x, 512, stride=2)
                x = Conv2D('conv2', x, 512, stride=1)
                x = Conv2D('conv3', x, 512, stride=2)
                x = Conv2D('conv4', x, 256, stride=1)
                x = FullyConnected('fc1', x, out_dim=1024, nl=BNReLU)
                x = FullyConnected('fc2', x, out_dim=512, nl=BNReLU)
                x = FullyConnected('fc3', x, out_dim=256, nl=BNReLU)
            return x

        @under_variable_scope()
        def colorization_network(mid_level_features, global_features):
            def upsample(x, factor=2):
                _, H, W, _ = x.get_shape().as_list()
                return tf.image.resize_nearest_neighbor(x, [2 * H, 2 * W])

            x = FusionLayer('fusion', mid_level_features, global_features)
            with argscope(Conv2D, kernel_shape=3, nl=BNReLU):
                x = Conv2D('conv1', x, 128, stride=1)
                x = upsample(x)
                x = Conv2D('conv2', x, 64, stride=1)
                x = Conv2D('conv3', x, 64, stride=1)
                x = upsample(x)
                x = Conv2D('conv4', x, 32, stride=1)
                x = Conv2D('conv5', x, 2, stride=1, nl=tf.tanh)
                x = upsample(x)
            return x

        @under_variable_scope()
        def classification_network(x, labels):
            x = FullyConnected('fc1', x, out_dim=365, nl=BNReLU)
            x = FullyConnected('fc2', x, out_dim=365, nl=tf.identity)
            return x

        with argscope(BatchNorm, use_local_stat=True):
            low_level_features = low_level(given_l / 50. - 1)
            mid_level_features = mid_level(low_level_features)
            if ctx.is_training:
                global_features = global_features(low_level_features)
            else:
                resized_given_rgb = tf.image.resize_images(given_rgb, [224, 224])
                resized_given_lab = rgb2lab(resized_given_rgb)
                resized_given_l = tf.expand_dims(resized_given_lab[:, :, :, 0], axis=-1)
                tmp = low_level(resized_given_l / 50. - 1)
                global_features = global_features(tmp)

        estimated_ab = colorization_network(
            mid_level_features, global_features) * 100.
        cls_logits = classification_network(global_features, labels)

        color_costs = tf.reduce_mean(tf.squared_difference(
            estimated_ab / 100., given_ab / 100.), name='color_costs')
        cls_costs = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=cls_logits, labels=labels), name='cls_costs')

        self.cost = tf.add(color_costs, ALPHA * cls_costs, name='total_costs')
        summary.add_moving_summary(self.cost, cls_costs, color_costs)
        tf.identity(lab2rgb(tf.concat([given_l, estimated_ab], axis=3)), name='prediction')
        with tf.name_scope('visualization'):
            estimated_rgb = lab2rgb(tf.concat([given_l, estimated_ab], axis=3))
            estimated_ab = lab2rgb(tf.concat([given_l * 0 + 50, estimated_ab], axis=3))
            given_ab = lab2rgb(tf.concat([given_l * 0 + 50, given_ab], axis=3))

            viz = tf.concat([given_rgb, estimated_rgb, given_ab, estimated_ab], 2)
            viz = tf.cast(tf.clip_by_value(viz, 0, 255), tf.uint8, name='viz')
            tf.summary.image('rgb(given, est), ab(given, est)', viz, max_outputs=max(30, BATCH_SIZE))

    def _get_optimizer(self):
        lr = symbolic_functions.get_scalar_var(
            'learning_rate', 5e-3, summary=True)
        return tf.train.AdamOptimizer(lr)


class ImageDecode(MapDataComponent):
    """Decode JPEG buffer to uint8 image array
    """

    def __init__(self, ds, mode='.jpg', dtype=np.uint8, index=0):
        def func(im_data):
            img = cv2.imdecode(np.asarray(
                bytearray(im_data), dtype=dtype), cv2.IMREAD_COLOR)
            return img[:, :, ::-1]
        super(ImageDecode, self).__init__(ds, func, index=index)


class RejectTooSmallImages(MapDataComponent):
    def __init__(self, ds, thresh=224, index=0):
        def func(img):
            h, w, _ = img.shape
            if (h < thresh) or (w < thresh):
                return None
            else:
                return img

        super(RejectTooSmallImages, self).__init__(ds, func, index=index)


class RejectGrayscaleImages(MapDataComponent):
    def __init__(self, ds, thresh=224, index=0):
        def func(img):
            h, w, c = img.shape
            if c == 1:
                return None
            if np.max((img[:, :, 0] - img[:, :, 1])**2) < 1:
                return None
            else:
                return img

        super(RejectGrayscaleImages, self).__init__(ds, func, index=index)


def get_data(train_lmdb, val_lmdb):
    augmentors = [imgaug.RandomCrop(224),
                  imgaug.Flip(horiz=True)]

    ds_train = LMDBDataPoint(train_lmdb, shuffle=True)
    ds_train = PrefetchData(ds_train, 100, 3)
    ds_train = ImageDecode(ds_train, index=0)
    ds_train = RejectTooSmallImages(ds_train, index=0)
    ds_train = AugmentImageComponent(ds_train, augmentors, index=0, copy=True)
    ds_train = RejectGrayscaleImages(ds_train, index=0)
    ds_train = BatchData(ds_train, BATCH_SIZE)
    ds_train = PrefetchDataZMQ(ds_train, nr_proc=12)

    augmentors = [imgaug.CenterCrop(224)]

    ds_val = LMDBDataPoint(val_lmdb, shuffle=False)
    ds_val = PrefetchData(ds_val, 100, 1)
    ds_val = ImageDecode(ds_val, index=0)
    ds_val = RejectTooSmallImages(ds_val, index=0)
    ds_val = AugmentImageComponent(ds_val, augmentors, index=0, copy=True)
    ds_val = RejectGrayscaleImages(ds_val, index=0)
    ds_val = FixedSizeData(ds_val, 500)
    ds_val = BatchData(ds_val, BATCH_SIZE)
    return ds_train, ds_val


def get_config(args):
    logger.auto_set_dir()

    ds_train, ds_val = get_data(args.train_lmdb, args.val_lmdb)

    return TrainConfig(
        model=Model(),
        dataflow=ds_train,
        callbacks=[
            ModelSaver(),
            InferenceRunner(ds_val, [ScalarStats('total_costs'), ScalarStats('cls_costs'), ScalarStats('color_costs')]),
        ],
        extra_callbacks=[
            MovingAverageSummary(),
            ProgressBar(['tower0/color_costs:0', 'tower0/cls_costs:0', 'tower0/total_costs:0']),
            MergeAllSummaries(),
            RunUpdateOps()
        ],
        steps_per_epoch=15000,
        max_epoch=240,
    )


def apply(args):
    im = cv2.imread(args.apply)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    im = np.stack([im, im, im], axis=-1)
    assert im is not None
    H, W, C = im.shape
    H = H // 8 * 8
    W = W // 8 * 8
    im = cv2.resize(im, (W, H))

    pred_config = PredictConfig(
        model=Model(H, W),
        session_init=get_model_loader(args.load),
        input_names=['rgb'],
        output_names=['prediction'])
    predictor = OfflinePredictor(pred_config)

    im = im[None, :, :, :].astype('float32')
    outputs = predictor(im)
    outputs = outputs[0][0, :, :, ::-1]
    cv2.imwrite(args.apply + '_input.jpg', im[0])
    cv2.imwrite(args.apply + '_output.jpg', outputs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--load', help='load model')
    parser.add_argument('--apply', help='run model on given image', default='')
    parser.add_argument('--train_lmdb', help='load model',
                        default='/scratch_shared/datasets/PLACE2/train_large_places365standard.lmdb')
    parser.add_argument('--val_lmdb', help='load model',
                        default='/scratch_shared/datasets/PLACE2/val_large.lmdb')
    args = parser.parse_args()

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if args.apply is not '':
        apply(args)
    else:
        config = get_config(args)

        if args.gpu:
            config.nr_tower = len(args.gpu.split(','))
        if args.load:
            config.session_init = SaverRestore(args.load)

        SyncMultiGPUTrainer(config).train()
