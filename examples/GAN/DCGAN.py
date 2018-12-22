#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: DCGAN.py
# Author: Yuxin Wu

import argparse
import glob
import numpy as np
import os
import tensorflow as tf

from tensorpack import *
from tensorpack.tfutils.scope_utils import auto_reuse_variable_scope
from tensorpack.utils.viz import stack_patches

from GAN import GANModelDesc, GANTrainer, RandomZData


"""
1. Download the 'aligned&cropped' version of CelebA dataset
   from http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

2. Start training:
    ./DCGAN-CelebA.py --data /path/to/img_align_celeba/ --crop-size 140
    Generated samples will be available through tensorboard

3. Visualize samples with an existing model:
    ./DCGAN-CelebA.py --load path/to/model --sample

You can also train on other images (just use any directory of jpg files in
`--data`). But you may need to change the preprocessing.

A pretrained model on CelebA is at http://models.tensorpack.com/GAN/
"""


class Model(GANModelDesc):
    def __init__(self, shape, batch, z_dim):
        self.shape = shape
        self.batch = batch
        self.zdim = z_dim

    def inputs(self):
        return [tf.placeholder(tf.float32, (None, self.shape, self.shape, 3), 'input')]

    def generator(self, z):
        """ return an image generated from z"""
        nf = 64
        l = FullyConnected('fc0', z, nf * 8 * 4 * 4, activation=tf.identity)
        l = tf.reshape(l, [-1, 4, 4, nf * 8])
        l = BNReLU(l)
        with argscope(Conv2DTranspose, activation=BNReLU, kernel_size=4, strides=2):
            l = Conv2DTranspose('deconv1', l, nf * 4)
            l = Conv2DTranspose('deconv2', l, nf * 2)
            l = Conv2DTranspose('deconv3', l, nf)
            l = Conv2DTranspose('deconv4', l, 3, activation=tf.identity)
            l = tf.tanh(l, name='gen')
        return l

    @auto_reuse_variable_scope
    def discriminator(self, imgs):
        """ return a (b, 1) logits"""
        nf = 64
        with argscope(Conv2D, kernel_size=4, strides=2):
            l = (LinearWrap(imgs)
                 .Conv2D('conv0', nf, activation=tf.nn.leaky_relu)
                 .Conv2D('conv1', nf * 2)
                 .BatchNorm('bn1')
                 .tf.nn.leaky_relu()
                 .Conv2D('conv2', nf * 4)
                 .BatchNorm('bn2')
                 .tf.nn.leaky_relu()
                 .Conv2D('conv3', nf * 8)
                 .BatchNorm('bn3')
                 .tf.nn.leaky_relu()
                 .FullyConnected('fct', 1)())
        return l

    def build_graph(self, image_pos):
        image_pos = image_pos / 128.0 - 1

        z = tf.random_uniform([self.batch, self.zdim], -1, 1, name='z_train')
        z = tf.placeholder_with_default(z, [None, self.zdim], name='z')

        with argscope([Conv2D, Conv2DTranspose, FullyConnected],
                      kernel_initializer=tf.truncated_normal_initializer(stddev=0.02)):
            with tf.variable_scope('gen'):
                image_gen = self.generator(z)
            tf.summary.image('generated-samples', image_gen, max_outputs=30)
            with tf.variable_scope('discrim'):
                vecpos = self.discriminator(image_pos)
                vecneg = self.discriminator(image_gen)

        self.build_losses(vecpos, vecneg)
        self.collect_variables()

    def optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=2e-4, trainable=False)
        return tf.train.AdamOptimizer(lr, beta1=0.5, epsilon=1e-3)


def get_augmentors():
    augs = []
    if args.load_size:
        augs.append(imgaug.Resize(args.load_size))
    if args.crop_size:
        augs.append(imgaug.CenterCrop(args.crop_size))
    augs.append(imgaug.Resize(args.final_size))
    return augs


def get_data():
    assert args.data
    imgs = glob.glob(args.data + '/*.jpg')
    ds = ImageFromFile(imgs, channel=3, shuffle=True)
    ds = AugmentImageComponent(ds, get_augmentors())
    ds = BatchData(ds, args.batch)
    ds = PrefetchDataZMQ(ds, 5)
    return ds


def sample(model, model_path, output_name='gen/gen'):
    pred = PredictConfig(
        session_init=get_model_loader(model_path),
        model=model,
        input_names=['z'],
        output_names=[output_name, 'z'])
    pred = SimpleDatasetPredictor(pred, RandomZData((100, args.z_dim)))
    for o in pred.get_result():
        o = o[0] + 1
        o = o * 128.0
        o = np.clip(o, 0, 255)
        o = o[:, :, :, ::-1]
        stack_patches(o, nr_row=10, nr_col=10, viz=True)


def get_args(default_batch=128, default_z_dim=100):
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--load', help='load model')
    parser.add_argument('--sample', action='store_true', help='view generated examples')
    parser.add_argument('--data', help='a jpeg directory')
    parser.add_argument('--load-size', help='size to load the original images', type=int)
    parser.add_argument('--crop-size', help='crop the original images', type=int)
    parser.add_argument(
        '--final-size', default=64, type=int,
        help='resize to this shape as inputs to network')
    parser.add_argument('--z-dim', help='hidden dimension', type=int, default=default_z_dim)
    parser.add_argument('--batch', help='batch size', type=int, default=default_batch)
    global args
    args = parser.parse_args()
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    return args


if __name__ == '__main__':
    args = get_args()
    M = Model(shape=args.final_size, batch=args.batch, z_dim=args.z_dim)
    if args.sample:
        sample(M, args.load)
    else:
        logger.auto_set_dir()
        GANTrainer(
            input=QueueInput(get_data()),
            model=M).train_with_defaults(
            callbacks=[ModelSaver()],
            steps_per_epoch=300,
            max_epoch=200,
            session_init=SaverRestore(args.load) if args.load else None
        )
