#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: hed.py
# Author: Yuxin Wu

import argparse
import numpy as np
import os
import cv2
import tensorflow as tf
from six.moves import zip

from tensorpack import *
from tensorpack.dataflow import dataset
from tensorpack.tfutils import gradproc, optimizer
from tensorpack.tfutils.summary import add_moving_summary, add_param_summary
from tensorpack.utils.gpu import get_num_gpu
from tensorpack.utils import logger


def class_balanced_sigmoid_cross_entropy(logits, label, name='cross_entropy_loss'):
    """
    The class-balanced cross entropy loss,
    as in `Holistically-Nested Edge Detection
    <http://arxiv.org/abs/1504.06375>`_.

    Args:
        logits: of shape (b, ...).
        label: of the same shape. the ground truth in {0,1}.
    Returns:
        class-balanced cross entropy loss.
    """
    with tf.name_scope('class_balanced_sigmoid_cross_entropy'):
        y = tf.cast(label, tf.float32)

        count_neg = tf.reduce_sum(1. - y)
        count_pos = tf.reduce_sum(y)
        beta = count_neg / (count_neg + count_pos)

        pos_weight = beta / (1 - beta)
        cost = tf.nn.weighted_cross_entropy_with_logits(logits=logits, targets=y, pos_weight=pos_weight)
        cost = tf.reduce_mean(cost * (1 - beta))
        zero = tf.equal(count_pos, 0.0)
    return tf.where(zero, 0.0, cost, name=name)


@layer_register(log_shape=True)
def CaffeBilinearUpSample(x, shape):
    """
    Deterministic bilinearly-upsample the input images.
    It is implemented by deconvolution with "BilinearFiller" in Caffe.
    It is aimed to mimic caffe behavior.

    Args:
        x (tf.Tensor): a NCHW tensor
        shape (int): the upsample factor

    Returns:
        tf.Tensor: a NCHW tensor.
    """
    inp_shape = x.shape.as_list()
    ch = inp_shape[1]
    assert ch == 1, "This layer only works for channel=1"

    shape = int(shape)
    filter_shape = 2 * shape

    def bilinear_conv_filler(s):
        """
        s: width, height of the conv filter
        https://github.com/BVLC/caffe/blob/99bd99795dcdf0b1d3086a8d67ab1782a8a08383/include/caffe/filler.hpp#L219-L268
        """
        f = np.ceil(float(s) / 2)
        c = float(2 * f - 1 - f % 2) / (2 * f)
        ret = np.zeros((s, s), dtype='float32')
        for x in range(s):
            for y in range(s):
                ret[x, y] = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
        return ret
    w = bilinear_conv_filler(filter_shape)
    w = np.repeat(w, ch * ch).reshape((filter_shape, filter_shape, ch, ch))

    weight_var = tf.constant(w, tf.float32,
                             shape=(filter_shape, filter_shape, ch, ch),
                             name='bilinear_upsample_filter')
    x = tf.pad(x, [[0, 0], [0, 0], [shape - 1, shape - 1], [shape - 1, shape - 1]], mode='SYMMETRIC')
    out_shape = tf.shape(x) * tf.constant([1, 1, shape, shape], tf.int32)
    deconv = tf.nn.conv2d_transpose(x, weight_var, out_shape,
                                    [1, 1, shape, shape], 'SAME', data_format='NCHW')
    edge = shape * (shape - 1)
    deconv = deconv[:, :, edge:-edge, edge:-edge]

    if inp_shape[2]:
        inp_shape[2] *= shape
    if inp_shape[3]:
        inp_shape[3] *= shape
    deconv.set_shape(inp_shape)
    return deconv


class Model(ModelDesc):
    def inputs(self):
        return [tf.placeholder(tf.float32, [None, None, None, 3], 'image'),
                tf.placeholder(tf.int32, [None, None, None], 'edgemap')]

    def build_graph(self, image, edgemap):
        image = image - tf.constant([104, 116, 122], dtype='float32')
        image = tf.transpose(image, [0, 3, 1, 2])
        edgemap = tf.expand_dims(edgemap, 3, name='edgemap4d')

        def branch(name, l, up):
            with tf.variable_scope(name):
                l = Conv2D('convfc', l, 1, kernel_size=1, activation=tf.identity,
                           use_bias=True,
                           kernel_initializer=tf.constant_initializer())
                while up != 1:
                    l = CaffeBilinearUpSample('upsample{}'.format(up), l, 2)
                    up = up // 2
                return l

        with argscope(Conv2D, kernel_size=3, activation=tf.nn.relu), \
                argscope([Conv2D, MaxPooling], data_format='NCHW'):
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

            final_map = Conv2D('convfcweight',
                               tf.concat([b1, b2, b3, b4, b5], 1), 1, kernel_size=1,
                               kernel_initializer=tf.constant_initializer(0.2),
                               use_bias=False, activation=tf.identity)
        costs = []
        for idx, b in enumerate([b1, b2, b3, b4, b5, final_map]):
            b = tf.transpose(b, [0, 2, 3, 1])
            output = tf.nn.sigmoid(b, name='output{}'.format(idx + 1))
            xentropy = class_balanced_sigmoid_cross_entropy(
                b, edgemap,
                name='xentropy{}'.format(idx + 1))
            costs.append(xentropy)

        # some magic threshold
        pred = tf.cast(tf.greater(output, 0.5), tf.int32, name='prediction')
        wrong = tf.cast(tf.not_equal(pred, edgemap), tf.float32)
        wrong = tf.reduce_mean(wrong, name='train_error')

        wd_w = tf.train.exponential_decay(2e-4, get_global_step_var(),
                                          80000, 0.7, True)
        wd_cost = tf.multiply(wd_w, regularize_cost('.*/W', tf.nn.l2_loss), name='wd_cost')
        costs.append(wd_cost)

        add_param_summary(('.*/W', ['histogram']))   # monitor W
        total_cost = tf.add_n(costs, name='cost')
        add_moving_summary(wrong, total_cost, *costs)
        return total_cost

    def optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=3e-5, trainable=False)
        opt = tf.train.AdamOptimizer(lr, epsilon=1e-3)
        return optimizer.apply_grad_processors(
            opt, [gradproc.ScaleGradient(
                [('convfcweight.*', 0.1), ('conv5_.*', 5)])])


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
            return img[h0:h0 + newh, w0:w0 + neww]

    if isTrain:
        shape_aug = [
            imgaug.RandomResize(xrange=(0.7, 1.5), yrange=(0.7, 1.5),
                                aspect_ratio_thres=0.15),
            imgaug.RotationAndCropValid(90),
            CropMultiple16(),
            imgaug.Flip(horiz=True),
            imgaug.Flip(vert=True)
        ]
    else:
        # the original image shape (321x481) in BSDS is not a multiple of 16
        IMAGE_SHAPE = (320, 480)
        shape_aug = [imgaug.CenterCrop(IMAGE_SHAPE)]
    ds = AugmentImageComponents(ds, shape_aug, (0, 1), copy=False)

    def f(m):   # thresholding
        m[m >= 0.50] = 1
        m[m < 0.50] = 0
        return m
    ds = MapDataComponent(ds, f, 1)

    if isTrain:
        augmentors = [
            imgaug.Brightness(63, clip=False),
            imgaug.Contrast((0.4, 1.5)),
        ]
        ds = AugmentImageComponent(ds, augmentors, copy=False)
        ds = BatchDataByShape(ds, 8, idx=0)
        ds = PrefetchDataZMQ(ds, 1)
    else:
        ds = BatchData(ds, 1)
    return ds


def view_data():
    ds = RepeatedData(get_data('train'), -1)
    ds.reset_state()
    for ims, edgemaps in ds:
        for im, edgemap in zip(ims, edgemaps):
            assert im.shape[0] % 16 == 0 and im.shape[1] % 16 == 0, im.shape
            cv2.imshow("im", im / 255.0)
            cv2.waitKey(1000)
            cv2.imshow("edge", edgemap)
            cv2.waitKey(1000)


def get_config():
    logger.auto_set_dir()
    dataset_train = get_data('train')
    steps_per_epoch = len(dataset_train) * 40
    dataset_val = get_data('val')

    return TrainConfig(
        dataflow=dataset_train,
        callbacks=[
            ModelSaver(),
            ScheduledHyperParamSetter('learning_rate', [(30, 6e-6), (45, 1e-6), (60, 8e-7)]),
            HumanHyperParamSetter('learning_rate'),
            InferenceRunner(dataset_val,
                            BinaryClassificationStats('prediction', 'edgemap4d'))
        ],
        model=Model(),
        steps_per_epoch=steps_per_epoch,
        max_epoch=100,
    )


def run(model_path, image_path, output):
    pred_config = PredictConfig(
        model=Model(),
        session_init=get_model_loader(model_path),
        input_names=['image'],
        output_names=['output' + str(k) for k in range(1, 7)])
    predictor = OfflinePredictor(pred_config)
    im = cv2.imread(image_path)
    assert im is not None
    im = cv2.resize(
        im, (im.shape[1] // 16 * 16, im.shape[0] // 16 * 16)
    )[None, :, :, :].astype('float32')
    outputs = predictor(im)
    if output is None:
        for k in range(6):
            pred = outputs[k][0]
            cv2.imwrite("out{}.png".format(
                '-fused' if k == 5 else str(k + 1)), pred * 255)
        logger.info("Results saved to out*.png")
    else:
        pred = outputs[5][0]
        cv2.imwrite(output, pred * 255)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--load', help='load model')
    parser.add_argument('--view', help='view dataset', action='store_true')
    parser.add_argument('--run', help='run model on images')
    parser.add_argument('--output', help='fused output filename. default to out-fused.png')
    args = parser.parse_args()
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if args.view:
        view_data()
    elif args.run:
        run(args.load, args.run, args.output)
    else:
        config = get_config()
        if args.load:
            config.session_init = get_model_loader(args.load)
        launch_train_with_config(
            config,
            SyncMultiGPUTrainer(max(get_num_gpu(), 1)))
