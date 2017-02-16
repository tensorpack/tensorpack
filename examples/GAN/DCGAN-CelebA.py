#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: DCGAN-CelebA.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import tensorflow as tf
import glob
import os, sys
import argparse

from tensorpack import *
from tensorpack.utils.viz import *
from tensorpack.tfutils.summary import add_moving_summary
from GAN import GANTrainer, RandomZData, GANModelDesc

"""
1. Download the 'aligned&cropped' version of CelebA dataset
   from http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
2. Start training:
    ./DCGAN-CelebA.py --data /path/to/image_align_celeba/
3. Visualize samples of a trained model:
    ./DCGAN-CelebA.py --load path/to/model --sample

You can also train on other images (just use any directory of jpg files in
`--data`). But you may need to change the preprocessing steps in `get_data()`.

A pretrained model on CelebA is at https://drive.google.com/open?id=0B9IPQTvr2BBkLUF2M0RXU1NYSkE
"""

SHAPE = 64
BATCH = 128
Z_DIM = 100


class Model(GANModelDesc):
    def _get_inputs(self):
        return [InputDesc(tf.float32, (None, SHAPE, SHAPE, 3), 'input')]

    def generator(self, z):
        """ return an image generated from z"""
        nf = 64
        l = FullyConnected('fc0', z, nf * 8 * 4 * 4, nl=tf.identity)
        l = tf.reshape(l, [-1, 4, 4, nf * 8])
        l = BNReLU(l)
        with argscope(Deconv2D, nl=BNReLU, kernel_shape=4, stride=2):
            l = Deconv2D('deconv1', l, [8, 8, nf * 4])
            l = Deconv2D('deconv2', l, [16, 16, nf * 2])
            l = Deconv2D('deconv3', l, [32, 32, nf])
            l = Deconv2D('deconv4', l, [64, 64, 3], nl=tf.identity)
            l = tf.tanh(l, name='gen')
        return l

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

        z = tf.random_uniform([BATCH, Z_DIM], -1, 1, name='z_train')
        z = tf.placeholder_with_default(z, [None, Z_DIM], name='z')

        with argscope([Conv2D, Deconv2D, FullyConnected],
                      W_init=tf.truncated_normal_initializer(stddev=0.02)):
            with tf.variable_scope('gen'):
                image_gen = self.generator(z)
            tf.summary.image('generated-samples', image_gen, max_outputs=30)
            with tf.variable_scope('discrim'):
                vecpos = self.discriminator(image_pos)
            with tf.variable_scope('discrim', reuse=True):
                vecneg = self.discriminator(image_gen)

        self.build_losses(vecpos, vecneg)
        self.collect_variables()

    def _get_optimizer(self):
        lr = symbolic_functions.get_scalar_var('learning_rate', 2e-4, summary=True)
        return tf.train.AdamOptimizer(lr, beta1=0.5, epsilon=1e-3)


def get_data(datadir):
    imgs = glob.glob(datadir + '/*.jpg')
    ds = ImageFromFile(imgs, channel=3, shuffle=True)
    augs = [imgaug.CenterCrop(140), imgaug.Resize(64)]
    ds = AugmentImageComponent(ds, augs)
    ds = BatchData(ds, BATCH)
    ds = PrefetchDataZMQ(ds, 1)
    return ds


def get_config():
    return TrainConfig(
        model=Model(),
        dataflow=get_data(args.data),
        callbacks=[ModelSaver()],
        session_config=get_default_sess_config(0.5),
        steps_per_epoch=300,
        max_epoch=200,
    )


def sample(model_path):
    pred = PredictConfig(
        session_init=get_model_loader(model_path),
        model=Model(),
        input_names=['z'],
        output_names=['gen/gen', 'z'])
    pred = SimpleDatasetPredictor(pred, RandomZData((100, 100)))
    for o in pred.get_result():
        o, zs = o[0] + 1, o[1]
        o = o * 128.0
        o = o[:, :, :, ::-1]
        viz = stack_patches(o, nr_row=10, nr_col=10, viz=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--load', help='load model')
    parser.add_argument('--sample', action='store_true', help='view generated examples')
    parser.add_argument('--data', help='a jpeg directory')
    args = parser.parse_args()
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    if args.sample:
        sample(args.load)
    else:
        assert args.data
        logger.auto_set_dir()
        config = get_config()
        if args.load:
            config.session_init = SaverRestore(args.load)
        GANTrainer(config).train()
