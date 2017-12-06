#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Patrick Wieschollek <mail@patwie.com>

import os
import argparse
import cv2
import six
import numpy as np
import tensorflow as tf

from tensorpack import *
from tensorpack.tfutils.scope_utils import auto_reuse_variable_scope
from tensorpack.tfutils.summary import add_moving_summary
from tensorpack.utils import logger
from data_sampler import ImageDecode
from GAN import MultiGPUGANTrainer, GANModelDesc
Reduction = tf.losses.Reduction


BATCH_SIZE = 6
CHANNELS = 3
SHAPE_LR = 32
NF = 64
VGG_MEAN = np.array([123.68, 116.779, 103.939])  # RGB


def normalize(v):
    assert isinstance(v, tf.Tensor)
    v.get_shape().assert_has_rank(4)
    dim = v.get_shape().as_list()
    return v / (dim[1] * dim[2] * dim[3])


def gram_matrix(v):
    assert isinstance(v, tf.Tensor)
    v.get_shape().assert_has_rank(4)
    dim = v.get_shape().as_list()
    v = normalize(v)
    v = tf.reshape(v, [-1, dim[1] * dim[2], dim[3]])
    return tf.matmul(v, v, transpose_a=True)


class Model(GANModelDesc):

    def __init__(self, height=SHAPE_LR, width=SHAPE_LR):
        super(Model, self).__init__()
        self.height = height
        self.width = width

    def _get_inputs(self):
        # mean-subtracted images
        return [InputDesc(tf.float32, (None, self.height * 1, self.width * 1, CHANNELS), 'Ilr'),
                InputDesc(tf.float32, (None, self.height * 4, self.width * 4, CHANNELS), 'Ihr')]

    def _build_graph(self, inputs):
        ctx = get_current_tower_context()
        Ilr, Ihr = inputs[0] / 255.0, inputs[1] / 255.0
        Ibicubic = tf.image.resize_bicubic(Ilr, [4 * self.height, 4 * self.width])

        def resnet_block(x, name):
            with tf.variable_scope(name):
                y = Conv2D('conv0', x, NF, nl=tf.nn.relu)
                y = Conv2D('conv1', y, NF, nl=tf.identity)
            return x + y

        def upsample(x, factor=2):
            _, h, w, _ = x.get_shape().as_list()
            x = tf.image.resize_nearest_neighbor(x, [factor * h, factor * w])
            return x

        def generator(x, Ibicubic):
            with argscope(Conv2D, kernel_shape=3, stride=1, nl=tf.nn.relu):
                x = Conv2D('conv1', x, NF)
                for i in range(10):
                    x = resnet_block(x, 'block_%i' % i)
                x = upsample(x)
                x = Conv2D('conv_post_1', x, NF)
                x = upsample(x)
                x = Conv2D('conv_post_2', x, NF)
                x = Conv2D('conv_post_3', x, NF)
                Ires = Conv2D('conv_post_4', x, 3, nl=tf.identity)
                Iest = tf.add(Ibicubic, Ires, name='Iest')
                return Iest

        @auto_reuse_variable_scope
        def discriminator(x):
            with argscope(LeakyReLU, alpha=0.2):
                with argscope(Conv2D, kernel_shape=3, stride=1, nl=LeakyReLU):
                    x = Conv2D('conv0', x, 32)
                    x = Conv2D('conv0b', x, 32, stride=2)
                    x = Conv2D('conv1', x, 64)
                    x = Conv2D('conv1b', x, 64, stride=2)
                    x = Conv2D('conv2', x, 128)
                    x = Conv2D('conv2b', x, 128, stride=2)
                    x = Conv2D('conv3', x, 256)
                    x = Conv2D('conv3b', x, 256, stride=2)
                    x = Conv2D('conv4', x, 512)
                    x = Conv2D('conv4b', x, 512, stride=2)

                x = FullyConnected('fc0', x, 1024, nl=LeakyReLU)
                x = FullyConnected('fc1', x, 1, nl=tf.identity)
            return x

        def additional_losses(a, b):
            with tf.variable_scope('VGG19'):
                x = tf.concat([a, b], axis=0)
                x = tf.reshape(x, [2 * BATCH_SIZE, 128, 128, 3])
                # VGG 19
                with varreplace.freeze_variables():
                    with argscope(Conv2D, kernel_shape=3, nl=tf.nn.relu):
                        conv1_1 = Conv2D('conv1_1', x, 64)
                        conv1_2 = Conv2D('conv1_2', conv1_1, 64)
                        pool1 = MaxPooling('pool1', conv1_2, 2)  # 64
                        conv2_1 = Conv2D('conv2_1', pool1, 128)
                        conv2_2 = Conv2D('conv2_2', conv2_1, 128)
                        pool2 = MaxPooling('pool2', conv2_2, 2)  # 32
                        conv3_1 = Conv2D('conv3_1', pool2, 256)
                        conv3_2 = Conv2D('conv3_2', conv3_1, 256)
                        conv3_3 = Conv2D('conv3_3', conv3_2, 256)
                        conv3_4 = Conv2D('conv3_4', conv3_3, 256)
                        pool3 = MaxPooling('pool3', conv3_4, 2)  # 16
                        conv4_1 = Conv2D('conv4_1', pool3, 512)
                        conv4_2 = Conv2D('conv4_2', conv4_1, 512)
                        conv4_3 = Conv2D('conv4_3', conv4_2, 512)
                        conv4_4 = Conv2D('conv4_4', conv4_3, 512)
                        pool4 = MaxPooling('pool4', conv4_4, 2)  # 8
                        conv5_1 = Conv2D('conv5_1', pool4, 512)
                        conv5_2 = Conv2D('conv5_2', conv5_1, 512)
                        conv5_3 = Conv2D('conv5_3', conv5_2, 512)
                        conv5_4 = Conv2D('conv5_4', conv5_3, 512)
                        pool5 = MaxPooling('pool5', conv5_4, 2)  # 4

            # perceptual loss
            with tf.name_scope('perceptual_loss'):
                phi_a_1, phi_b_1 = tf.split(pool2, 2, axis=0)
                phi_a_2, phi_b_2 = tf.split(pool5, 2, axis=0)

                logger.info('Create perceptual loss for layer {} with shape {}'.format(pool2.name, pool2.get_shape()))
                pool2_loss = tf.losses.mean_squared_error(phi_a_1, phi_b_1, reduction=Reduction.MEAN)
                logger.info('Create perceptual loss for layer {} with shape {}'.format(pool5.name, pool5.get_shape()))
                pool5_loss = tf.losses.mean_squared_error(phi_a_2, phi_b_2, reduction=Reduction.MEAN)

            # texture loss
            with tf.name_scope('texture_loss'):
                def texture_loss(x, p=16):
                    x = normalize(x)
                    _, h, w, c = x.get_shape().as_list()
                    assert h % p == 0 and w % p == 0
                    logger.info('Create texture loss for layer {} with shape {}'.format(x.name, x.get_shape()))

                    x = tf.space_to_batch_nd(x, [p, p], [[0, 0], [0, 0]])
                    x = tf.reshape(x, [p, p, -1, h // p, w // p, c])
                    x = tf.transpose(x, [2, 3, 4, 0, 1, 5])
                    patches_a, patches_b = tf.split(x, 2)   # each is b,h/p,w/p,p,p,c

                    patches_a = tf.reshape(patches_a, [-1, p, p, c])
                    patches_b = tf.reshape(patches_b, [-1, p, p, c])
                    return tf.losses.mean_squared_error(
                        gram_matrix(patches_a),
                        gram_matrix(patches_b),
                        reduction=Reduction.SUM
                    ) * (1.0 / BATCH_SIZE)

                texture_loss_conv1_1 = tf.identity(texture_loss(conv1_1), name='normalized_conv1_1')
                texture_loss_conv2_1 = tf.identity(texture_loss(conv2_1), name='normalized_conv2_1')
                texture_loss_conv3_1 = tf.identity(texture_loss(conv3_1), name='normalized_conv3_1')

            return [pool2_loss, pool5_loss, texture_loss_conv1_1, texture_loss_conv2_1, texture_loss_conv3_1]

        with tf.variable_scope('gen'):
            fake_hr = generator(Ilr, Ibicubic)
            real_hr = Ihr

        VGG_MEAN_TENSOR = tf.constant(VGG_MEAN, dtype=tf.float32)
        tf.add(fake_hr, VGG_MEAN_TENSOR / 255.0, name='prediction')

        if ctx.is_training:
            with tf.variable_scope('discrim'):
                real_score = discriminator(real_hr)
                fake_score = discriminator(fake_hr)

            self.build_losses(real_score, fake_score)

            additional_losses = additional_losses(fake_hr, real_hr)
            with tf.name_scope('additional_losses'):
                # see table 2 from appendix
                loss = []
                loss.append(tf.multiply(1., self.g_loss, name="loss_LA"))
                loss.append(tf.multiply(2e-1, additional_losses[0], name="loss_LP1"))
                loss.append(tf.multiply(2e-2, additional_losses[1], name="loss_LP2"))
                loss.append(tf.multiply(3e-7, additional_losses[2], name="loss_LT1"))
                loss.append(tf.multiply(1e-6, additional_losses[3], name="loss_LT2"))
                loss.append(tf.multiply(1e-6, additional_losses[4], name="loss_LT3"))

            self.g_loss = self.g_loss + tf.add_n(loss, name='total_g_loss')
            add_moving_summary(self.g_loss, *loss)

            # visualization
            viz = (tf.concat([Ibicubic, fake_hr, real_hr], 2)) * 255. + VGG_MEAN_TENSOR
            viz = tf.cast(tf.clip_by_value(viz, 0, 255), tf.uint8, name='viz')
            tf.summary.image('input,fake,real', viz,
                             max_outputs=max(30, BATCH_SIZE))

            self.collect_variables()

    def _get_optimizer(self):
        lr = tf.get_variable(
            'learning_rate', initializer=1e-4, trainable=False)
        opt = tf.train.AdamOptimizer(lr)
        gradprocs = [gradproc.ScaleGradient([('discrim/*', 0.3)])]
        return optimizer.apply_grad_processors(opt, gradprocs)


def apply(model_path, lowres_path="", output_path='.'):
    assert os.path.isfile(lowres_path)
    assert os.path.isdir(output_path)
    lr = cv2.imread(lowres_path).astype(np.float32)
    baseline = cv2.resize(lr, (0, 0), fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    LR_SIZE_H, LR_SIZE_W = lr.shape[:2]
    lr -= VGG_MEAN

    predict_func = OfflinePredictor(PredictConfig(
        model=Model(LR_SIZE_H, LR_SIZE_W),
        session_init=get_model_loader(model_path),
        input_names=['Ilr'],
        output_names=['prediction']))

    pred = predict_func(lr[None, ...])
    p = np.clip(pred[0][0, ...] * 255, 0, 255)

    cv2.imwrite(os.path.join(output_path, "predition.png"), p)
    cv2.imwrite(os.path.join(output_path, "baseline.png"), baseline)


def get_data(lmdb):
    ds = LMDBDataPoint(lmdb, shuffle=True)
    ds = ImageDecode(ds, index=0)
    augmentors = [imgaug.RandomCrop(128),
                  imgaug.Flip(horiz=True)]
    ds = AugmentImageComponent(ds, augmentors, index=0, copy=True)
    ds = MapData(ds, lambda x: x - VGG_MEAN)
    ds = MapData(ds, lambda x: [cv2.resize(x[0], (32, 32), interpolation=cv2.INTER_AREA), x[0]])
    ds = PrefetchDataZMQ(ds, 8)
    ds = BatchData(ds, BATCH_SIZE)
    return ds


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--load', help='load model')
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--lmdb', help='path to lmdb_file')
    parser.add_argument('--vgg19', help='load model', default="")
    parser.add_argument('--lowres', help='low resolution image as input', default="", type=str)
    parser.add_argument('--output', help='directory for saving predicted high-res image', default=".", type=str)
    args = parser.parse_args()

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if args.apply:
        apply(args.load, args.lowres, args.output)
    else:
        logger.auto_set_dir()

        if args.load:
            session_init = SaverRestore(args.load)
        else:
            assert os.path.isfile(args.vgg19)
            param_dict = np.load(args.vgg19, encoding='latin1').item()
            param_dict = {'VGG19/' + name: value for name, value in six.iteritems(param_dict)}
            session_init = DictRestore(param_dict)

        nr_tower = max(get_nr_gpu(), 1)
        data = QueueInput(get_data(args.lmdb))
        model = Model()
        if nr_tower == 1:
            trainer = GANTrainer(data, model)
        else:
            trainer = MultiGPUGANTrainer(nr_tower, data, model)

        trainer.train_with_defaults(
            callbacks=[
                ModelSaver(keep_checkpoint_every_n_hours=2)
            ],
            session_init=session_init,
            steps_per_epoch=data.size(),
            max_epoch=2000
        )
