#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: Image2Image.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import numpy as np
import tensorflow as tf
import glob
import pickle
import os
import sys
import argparse
import cv2

from tensorpack import *
from tensorpack.utils.viz import *
from tensorpack.tfutils.summary import add_moving_summary
import tensorpack.tfutils.symbolic_functions as symbf
from GAN import GANTrainer, GANModelDesc

"""
To train Image-to-Image translation model with image pairs:
    ./Image2Image.py --data /path/to/datadir --mode {AtoB,BtoA}
    # datadir should contain jpg images of shpae 2s x s, formed by A and B
    # you can download some data from the original authors:
    # https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/

Speed:
    On GTX1080 with BATCH=1, the speed is about 9.3it/s (the original torch version is 9.5it/s)

Training visualization will appear be in tensorboard.
To visualize on test set:
    ./Image2Image.py --sample --data /path/to/test/datadir --mode {AtoB,BtoA} --load model

"""

SHAPE = 256
BATCH = 1
IN_CH = 3
OUT_CH = 3
LAMBDA = 100
NF = 64  # number of filter


class Model(GANModelDesc):
    def _get_inputs(self):
        return [InputDesc(tf.float32, (None, SHAPE, SHAPE, IN_CH), 'input'),
                InputDesc(tf.float32, (None, SHAPE, SHAPE, OUT_CH), 'output')]

    def generator(self, imgs):
        # imgs: input: 256x256xch
        # U-Net structure, it's slightly different from the original on the location of relu/lrelu
        with argscope(BatchNorm, use_local_stat=True), \
                argscope(Dropout, is_training=True):
            # always use local stat for BN, and apply dropout even in testing
            with argscope(Conv2D, kernel_shape=4, stride=2,
                          nl=lambda x, name: LeakyReLU(BatchNorm('bn', x), name=name)):
                e1 = Conv2D('conv1', imgs, NF, nl=LeakyReLU)
                e2 = Conv2D('conv2', e1, NF * 2)
                e3 = Conv2D('conv3', e2, NF * 4)
                e4 = Conv2D('conv4', e3, NF * 8)
                e5 = Conv2D('conv5', e4, NF * 8)
                e6 = Conv2D('conv6', e5, NF * 8)
                e7 = Conv2D('conv7', e6, NF * 8)
                e8 = Conv2D('conv8', e7, NF * 8, nl=BNReLU)  # 1x1
            with argscope(Deconv2D, nl=BNReLU, kernel_shape=4, stride=2):
                return (LinearWrap(e8)
                        .Deconv2D('deconv1', NF * 8)
                        .Dropout()
                        .ConcatWith(e7, 3)
                        .Deconv2D('deconv2', NF * 8)
                        .Dropout()
                        .ConcatWith(e6, 3)
                        .Deconv2D('deconv3', NF * 8)
                        .Dropout()
                        .ConcatWith(e5, 3)
                        .Deconv2D('deconv4', NF * 8)
                        .ConcatWith(e4, 3)
                        .Deconv2D('deconv5', NF * 4)
                        .ConcatWith(e3, 3)
                        .Deconv2D('deconv6', NF * 2)
                        .ConcatWith(e2, 3)
                        .Deconv2D('deconv7', NF * 1)
                        .ConcatWith(e1, 3)
                        .Deconv2D('deconv8', OUT_CH, nl=tf.tanh)())

    def discriminator(self, inputs, outputs):
        """ return a (b, 1) logits"""
        l = tf.concat([inputs, outputs], 3)
        with argscope(Conv2D, nl=tf.identity, kernel_shape=4, stride=2):
            l = (LinearWrap(l)
                 .Conv2D('conv0', NF, nl=LeakyReLU)
                 .Conv2D('conv1', NF * 2)
                 .BatchNorm('bn1').LeakyReLU()
                 .Conv2D('conv2', NF * 4)
                 .BatchNorm('bn2').LeakyReLU()
                 .Conv2D('conv3', NF * 8, stride=1, padding='VALID')
                 .BatchNorm('bn3').LeakyReLU()
                 .Conv2D('convlast', 1, stride=1, padding='VALID')())
        return l

    def _build_graph(self, inputs):
        input, output = inputs
        input, output = input / 128.0 - 1, output / 128.0 - 1

        with argscope([Conv2D, Deconv2D],
                      W_init=tf.truncated_normal_initializer(stddev=0.02)), \
                argscope(LeakyReLU, alpha=0.2):
            with tf.variable_scope('gen'):
                fake_output = self.generator(input)
            with tf.variable_scope('discrim'):
                real_pred = self.discriminator(input, output)
            with tf.variable_scope('discrim', reuse=True):
                fake_pred = self.discriminator(input, fake_output)

        self.build_losses(real_pred, fake_pred)
        errL1 = tf.reduce_mean(tf.abs(fake_output - output), name='L1_loss')
        self.g_loss = tf.add(self.g_loss, LAMBDA * errL1, name='total_g_loss')
        add_moving_summary(errL1, self.g_loss)

        # tensorboard visualization
        if IN_CH == 1:
            input = tf.image.grayscale_to_rgb(input)
        if OUT_CH == 1:
            output = tf.image.grayscale_to_rgb(output)
            fake_output = tf.image.grayscale_to_rgb(fake_output)
        viz = (tf.concat([input, output, fake_output], 2) + 1.0) * 128.0
        viz = tf.cast(tf.clip_by_value(viz, 0, 255), tf.uint8, name='viz')
        tf.summary.image('input,output,fake', viz, max_outputs=max(30, BATCH))

        self.collect_variables()

    def _get_optimizer(self):
        lr = symbolic_functions.get_scalar_var('learning_rate', 2e-4, summary=True)
        return tf.train.AdamOptimizer(lr, beta1=0.5, epsilon=1e-3)


def split_input(img):
    """
    img: an RGB image of shape (s, 2s, 3).
    :return: [input, output]
    """
    # split the image into left + right pairs
    s = img.shape[0]
    assert img.shape[1] == 2 * s
    input, output = img[:, :s, :], img[:, s:, :]
    if args.mode == 'BtoA':
        input, output = output, input
    if IN_CH == 1:
        input = cv2.cvtColor(input, cv2.COLOR_RGB2GRAY)[:, :, np.newaxis]
    if OUT_CH == 1:
        output = cv2.cvtColor(output, cv2.COLOR_RGB2GRAY)[:, :, np.newaxis]
    return [input, output]


def get_data():
    datadir = args.data
    imgs = glob.glob(os.path.join(datadir, '*.jpg'))
    ds = ImageFromFile(imgs, channel=3, shuffle=True)

    ds = MapData(ds, lambda dp: split_input(dp[0]))
    assert SHAPE < 286  # this is the parameter used in the paper
    augs = [imgaug.Resize(286), imgaug.RandomCrop(SHAPE)]
    ds = AugmentImageComponents(ds, augs, (0, 1))
    ds = BatchData(ds, BATCH)
    ds = PrefetchData(ds, 100, 1)
    return ds


def get_config():
    logger.auto_set_dir()
    dataset = get_data()
    return TrainConfig(
        dataflow=dataset,
        callbacks=[
            PeriodicTrigger(ModelSaver(), every_k_epochs=3),
            ScheduledHyperParamSetter('learning_rate', [(200, 1e-4)])
        ],
        model=Model(),
        steps_per_epoch=dataset.size(),
        max_epoch=300,
    )


def sample(datadir, model_path):
    pred = PredictConfig(
        session_init=get_model_loader(model_path),
        model=Model(),
        input_names=['input', 'output'],
        output_names=['viz'])

    imgs = glob.glob(os.path.join(datadir, '*.jpg'))
    ds = ImageFromFile(imgs, channel=3, shuffle=True)
    ds = BatchData(MapData(ds, lambda dp: split_input(dp[0])), 6)

    pred = SimpleDatasetPredictor(pred, ds)
    for o in pred.get_result():
        o = o[0][:, :, :, ::-1]
        stack_patches(o, nr_row=3, nr_col=2, viz=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--load', help='load model')
    parser.add_argument('--sample', action='store_true', help='run sampling')
    parser.add_argument('--data', help='Image directory')
    parser.add_argument('--mode', choices=['AtoB', 'BtoA'], default='AtoB')
    parser.add_argument('-b', '--batch', type=int, default=1)
    global args
    args = parser.parse_args()
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    assert args.data

    BATCH = args.batch

    if args.sample:
        sample(args.data, args.load)
    else:
        config = get_config()
        if args.load:
            config.session_init = SaverRestore(args.load)
        GANTrainer(config).train()
