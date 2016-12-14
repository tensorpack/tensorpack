#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: Image2Image.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import numpy as np
import tensorflow as tf
import glob, pickle
import os, sys
import argparse
import cv2

from tensorpack import *
from tensorpack.utils.viz import *
from tensorpack.tfutils.summary import add_moving_summary, summary_moving_average
import tensorpack.tfutils.symbolic_functions as symbf
from GAN import GANTrainer, RandomZData, build_GAN_losses

"""
To train:
    ./Image2Image.py --data /path/to/datadir --mode {AtoB,BtoA}
    # datadir should contain images of shpae 2s x s, formed by A and B
    # you can download some data from the original pix2pix repo: https://github.com/phillipi/pix2pix#datasets
    # training visualization will appear be in tensorboard

To visualize on test set:
    ./Image2Image.py --sample --data /path/to/test/datadir --mode {AtoB,BtoA} --load pretrained.model
"""

SHAPE = 256
BATCH = 4
IN_CH = 3
OUT_CH = 3
LAMBDA = 100
NF = 64 # number of filter

class Model(ModelDesc):
    def _get_input_vars(self):
        return [InputVar(tf.float32, (None, SHAPE, SHAPE, IN_CH), 'input') ,
                InputVar(tf.float32, (None, SHAPE, SHAPE, OUT_CH), 'output') ]

    def generator(self, imgs):
        # imgs: input: 256x256xch
        # U-Net structure, slightly different from the original on the location of relu/lrelu
        with argscope(BatchNorm, use_local_stat=True), \
                argscope(Dropout, is_training=True):
            # always use local stat for BN, and apply dropout even in testing
            with argscope(Conv2D, kernel_shape=4, stride=2,
                    nl=lambda x, name: LeakyReLU(BatchNorm('bn', x), name=name)):
                e1 = Conv2D('conv1', imgs, NF, nl=LeakyReLU)
                e2 = Conv2D('conv2', e1, NF*2)
                e3 = Conv2D('conv3', e2, NF*4)
                e4 = Conv2D('conv4', e3, NF*8)
                e5 = Conv2D('conv5', e4, NF*8)
                e6 = Conv2D('conv6', e5, NF*8)
                e7 = Conv2D('conv7', e6, NF*8)
                e8 = Conv2D('conv8', e7, NF*8, nl=BNReLU)  # 1x1
            with argscope(Deconv2D, nl=BNReLU, kernel_shape=4, stride=2):
                return (LinearWrap(e8)
                    .Deconv2D('deconv1', NF*8)
                    .Dropout()
                    .ConcatWith(3, e7)
                    .Deconv2D('deconv2', NF*8)
                    .Dropout()
                    .ConcatWith(3, e6)
                    .Deconv2D('deconv3', NF*8)
                    .Dropout()
                    .ConcatWith(3, e5)
                    .Deconv2D('deconv4', NF*8)
                    .ConcatWith(3, e4)
                    .Deconv2D('deconv5', NF*4)
                    .ConcatWith(3, e3)
                    .Deconv2D('deconv6', NF*2)
                    .ConcatWith(3, e2)
                    .Deconv2D('deconv7', NF*1)
                    .ConcatWith(3, e1)
                    .Deconv2D('deconv8', OUT_CH, nl=tf.tanh)())

    def discriminator(self, inputs, outputs):
        """ return a (b, 1) logits"""
        l = tf.concat(3, [inputs, outputs])
        with argscope(Conv2D, nl=tf.identity, kernel_shape=4, stride=2):
            l = (LinearWrap(l)
                .Conv2D('conv0', NF, nl=LeakyReLU)
                .Conv2D('conv1', NF*2)
                .BatchNorm('bn1').LeakyReLU()
                .Conv2D('conv2', NF*4)
                .BatchNorm('bn2').LeakyReLU()
                .Conv2D('conv3', NF*8, stride=1, padding='VALID')
                .BatchNorm('bn3').LeakyReLU()
                .Conv2D('convlast', 1, stride=1, padding='VALID')())
        return l

    def _build_graph(self, input_vars):
        input, output = input_vars
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

        self.g_loss, self.d_loss = build_GAN_losses(real_pred, fake_pred)
        errL1 = tf.reduce_mean(tf.abs(fake_output - output), name='L1_loss')
        self.g_loss = tf.add(self.g_loss, LAMBDA * errL1, name='total_g_loss')
        add_moving_summary(errL1, self.g_loss)

        # tensorboard visualization
        if IN_CH == 1:
            input = tf.image.grayscale_to_rgb(input)
        if OUT_CH == 1:
            output = tf.image.grayscale_to_rgb(output)
            fake_output = tf.image.grayscale_to_rgb(fake_output)
        viz = (tf.concat(2, [input, output, fake_output]) + 1.0) * 128.0
        viz = tf.cast(tf.clip_by_value(viz, 0, 255), tf.uint8, name='viz')
        tf.image_summary('gen', viz, max_outputs=max(30, BATCH))

        all_vars = tf.trainable_variables()
        self.g_vars = [v for v in all_vars if v.name.startswith('gen/')]
        self.d_vars = [v for v in all_vars if v.name.startswith('discrim/')]

def split_input(img):
    """
    img: an image with shape (s, 2s, 3)
    :return: [input, output]
    """
    s = img.shape[0]
    input, output = img[:,:s,:], img[:,s:,:]
    if args.mode == 'BtoA':
        input, output = output, input
    if IN_CH == 1:
        input = cv2.cvtColor(input, cv2.COLOR_RGB2GRAY)
    if OUT_CH == 1:
        output = cv2.cvtColor(output, cv2.COLOR_RGB2GRAY)
    return [input, output]

def get_data():
    datadir = args.data
    # assume each image is 512x256 split to left and right
    imgs = glob.glob(os.path.join(datadir, '*.jpg'))
    ds = ImageFromFile(imgs, channel=3, shuffle=True)
    ds = MapData(ds, lambda dp: split_input(dp[0]))
    augs = [ imgaug.Resize(286), imgaug.RandomCrop(256) ]
    ds = AugmentImageComponents(ds, augs, (0, 1))
    ds = BatchData(ds, BATCH)
    ds = PrefetchDataZMQ(ds, 1)
    return ds

def get_config():
    logger.auto_set_dir()
    dataset = get_data()
    lr = symbolic_functions.get_scalar_var('learning_rate', 2e-4, summary=True)
    return TrainConfig(
        dataset=dataset,
        optimizer=tf.train.AdamOptimizer(lr, beta1=0.5, epsilon=1e-3),
        callbacks=Callbacks([
            StatPrinter(), ModelSaver(),
            ScheduledHyperParamSetter('learning_rate', [(200, 1e-4)])
        ]),
        model=Model(),
        step_per_epoch=dataset.size(),
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
        o = o[0][:,:,:,::-1]
        viz = next(build_patch_list(o, nr_row=3, nr_col=2, viz=True))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--load', help='load model')
    parser.add_argument('--sample', action='store_true', help='run sampling')
    parser.add_argument('--data', help='Image directory')
    parser.add_argument('--mode', choices=['AtoB', 'BtoA'], default='AtoB')
    global args
    args = parser.parse_args()
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    if args.sample:
        sample(args.data, args.load)
    else:
        assert args.data
        config = get_config()
        if args.load:
            config.session_init = SaverRestore(args.load)
        GANTrainer(config).train()
