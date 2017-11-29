#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: DCGAN.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import glob
import numpy as np
import os
import argparse


from tensorpack import *
from tensorpack.utils.viz import stack_patches
from tensorpack.tfutils.scope_utils import auto_reuse_variable_scope
from tensorpack.utils.globvars import globalns as opt
import tensorflow as tf

from GAN import GANTrainer, RandomZData, GANModelDesc

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

# global vars
opt.SHAPE = 64
opt.BATCH = 128
opt.Z_DIM = 100


class Model(GANModelDesc):
    def _get_inputs(self):
        return [InputDesc(tf.float32, (None, opt.SHAPE, opt.SHAPE, 3), 'input')]

    def generator(self, z):
        """ return an image generated from z"""
        nf = 64
        l = FullyConnected('fc0', z, nf * 8 * 4 * 4, nl=tf.identity)
        l = tf.reshape(l, [-1, 4, 4, nf * 8])
        l = BNReLU(l)
        with argscope(Deconv2D, nl=BNReLU, kernel_shape=4, stride=2):
            l = Deconv2D('deconv1', l, nf * 4)
            l = Deconv2D('deconv2', l, nf * 2)
            l = Deconv2D('deconv3', l, nf)
            l = Deconv2D('deconv4', l, 3, nl=tf.identity)
            l = tf.tanh(l, name='gen')
        return l

    @auto_reuse_variable_scope
    def discriminator(self, imgs):
        """ return a (b, 1) logits"""
        nf = 64
        with argscope(Conv2D, nl=tf.identity, kernel_shape=4, stride=2), \
                argscope(LeakyReLU, alpha=0.2):
            l = (LinearWrap(imgs)
                 .Conv2D('conv0', nf, nl=LeakyReLU)
                 .Conv2D('conv1', nf * 2)
                 .BatchNorm('bn1').LeakyReLU()
                 .Conv2D('conv2', nf * 4)
                 .BatchNorm('bn2').LeakyReLU()
                 .Conv2D('conv3', nf * 8)
                 .BatchNorm('bn3').LeakyReLU()
                 .FullyConnected('fct', 1, nl=tf.identity)())
        return l

    def _build_graph(self, inputs):
        image_pos = inputs[0]
        image_pos = image_pos / 128.0 - 1

        z = tf.random_uniform([opt.BATCH, opt.Z_DIM], -1, 1, name='z_train')
        z = tf.placeholder_with_default(z, [None, opt.Z_DIM], name='z')

        with argscope([Conv2D, Deconv2D, FullyConnected],
                      W_init=tf.truncated_normal_initializer(stddev=0.02)):
            with tf.variable_scope('gen'):
                image_gen = self.generator(z)
            tf.summary.image('generated-samples', image_gen, max_outputs=30)
            with tf.variable_scope('discrim'):
                vecpos = self.discriminator(image_pos)
                vecneg = self.discriminator(image_gen)

        self.build_losses(vecpos, vecneg)
        self.collect_variables()

    def _get_optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=2e-4, trainable=False)
        return tf.train.AdamOptimizer(lr, beta1=0.5, epsilon=1e-3)


def get_augmentors():
    augs = []
    if opt.load_size:
        augs.append(imgaug.Resize(opt.load_size))
    if opt.crop_size:
        augs.append(imgaug.CenterCrop(opt.crop_size))
    augs.append(imgaug.Resize(opt.SHAPE))
    return augs


def get_data(datadir):
    imgs = glob.glob(datadir + '/*.jpg')
    ds = ImageFromFile(imgs, channel=3, shuffle=True)
    ds = AugmentImageComponent(ds, get_augmentors())
    ds = BatchData(ds, opt.BATCH)
    ds = PrefetchDataZMQ(ds, 5)
    return ds


def sample(model, model_path, output_name='gen/gen'):
    pred = PredictConfig(
        session_init=get_model_loader(model_path),
        model=model,
        input_names=['z'],
        output_names=[output_name, 'z'])
    pred = SimpleDatasetPredictor(pred, RandomZData((100, opt.Z_DIM)))
    for o in pred.get_result():
        o = o[0] + 1
        o = o * 128.0
        o = np.clip(o, 0, 255)
        o = o[:, :, :, ::-1]
        stack_patches(o, nr_row=10, nr_col=10, viz=True)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--load', help='load model')
    parser.add_argument('--sample', action='store_true', help='view generated examples')
    parser.add_argument('--data', help='a jpeg directory')
    parser.add_argument('--load-size', help='size to load the original images', type=int)
    parser.add_argument('--crop-size', help='crop the original images', type=int)
    args = parser.parse_args()
    opt.use_argument(args)
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    return args


if __name__ == '__main__':
    args = get_args()
    if args.sample:
        sample(Model(), args.load)
    else:
        assert args.data
        logger.auto_set_dir()
        GANTrainer(
            input=QueueInput(get_data(args.data)),
            model=Model()).train_with_defaults(
            callbacks=[ModelSaver()],
            steps_per_epoch=300,
            max_epoch=200,
            session_init=SaverRestore(args.load) if args.load else None
        )
