#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: GAN.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import tensorflow as tf
import numpy as np
import time
from tensorpack import (FeedfreeTrainerBase, QueueInput,
                        ModelDesc, DataFlow, StagingInputWrapper,
                        MultiGPUTrainerBase, LeastLoadedDeviceSetter)
from tensorpack.tfutils.summary import add_moving_summary


class GANModelDesc(ModelDesc):
    def collect_variables(self, g_scope='gen', d_scope='discrim'):
        """
        Assign self.g_vars to the parameters under scope `g_scope`,
        and same with self.d_vars.
        """
        self.g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, g_scope)
        assert self.g_vars
        self.d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, d_scope)
        assert self.d_vars

    def build_losses(self, logits_real, logits_fake):
        """D and G play two-player minimax game with value function V(G,D)

          min_G max _D V(D, G) = IE_{x ~ p_data} [log D(x)] + IE_{z ~ p_fake} [log (1 - D(G(z)))]

        Args:
            logits_real (tf.Tensor): discrim logits from real samples
            logits_fake (tf.Tensor): discrim logits from fake samples produced by generator
        """
        with tf.name_scope("GAN_loss"):
            score_real = tf.sigmoid(logits_real)
            score_fake = tf.sigmoid(logits_fake)
            tf.summary.histogram('score-real', score_real)
            tf.summary.histogram('score-fake', score_fake)

            with tf.name_scope("discrim"):
                d_loss_pos = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=logits_real, labels=tf.ones_like(logits_real)), name='loss_real')
                d_loss_neg = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=logits_fake, labels=tf.zeros_like(logits_fake)), name='loss_fake')

                d_pos_acc = tf.reduce_mean(tf.cast(score_real > 0.5, tf.float32), name='accuracy_real')
                d_neg_acc = tf.reduce_mean(tf.cast(score_fake < 0.5, tf.float32), name='accuracy_fake')

                d_accuracy = tf.add(.5 * d_pos_acc, .5 * d_neg_acc, name='accuracy')
                self.d_loss = tf.add(.5 * d_loss_pos, .5 * d_loss_neg, name='loss')

            with tf.name_scope("gen"):
                self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=logits_fake, labels=tf.ones_like(logits_fake)), name='loss')
                g_accuracy = tf.reduce_mean(tf.cast(score_fake > 0.5, tf.float32), name='accuracy')

            add_moving_summary(self.g_loss, self.d_loss, d_accuracy, g_accuracy)


class GANTrainer(FeedfreeTrainerBase):
    def __init__(self, config):
        self._input_source = QueueInput(config.dataflow)
        super(GANTrainer, self).__init__(config)

    def _setup(self):
        super(GANTrainer, self)._setup()
        self.build_train_tower()
        opt = self.model.get_optimizer()

        # by default, run one d_min after one g_min
        self.g_min = opt.minimize(self.model.g_loss, var_list=self.model.g_vars, name='g_op')
        with tf.control_dependencies([self.g_min]):
            self.d_min = opt.minimize(self.model.d_loss, var_list=self.model.d_vars, name='d_op')
        self.train_op = self.d_min


class SeparateGANTrainer(FeedfreeTrainerBase):
    """ A GAN trainer which runs two optimization ops with a certain ratio, one in each step. """
    def __init__(self, config, d_period=1, g_period=1):
        """
        Args:
            d_period(int): period of each d_opt run
            g_period(int): period of each g_opt run
        """
        self._input_source = QueueInput(config.dataflow)
        self._d_period = int(d_period)
        self._g_period = int(g_period)
        assert min(d_period, g_period) == 1
        super(SeparateGANTrainer, self).__init__(config)

    def _setup(self):
        super(SeparateGANTrainer, self)._setup()
        self.build_train_tower()

        opt = self.model.get_optimizer()
        self.d_min = opt.minimize(
            self.model.d_loss, var_list=self.model.d_vars, name='d_min')
        self.g_min = opt.minimize(
            self.model.g_loss, var_list=self.model.g_vars, name='g_min')
        self._cnt = 1

    def run_step(self):
        if self._cnt % (self._d_period) == 0:
            self.hooked_sess.run(self.d_min)
        if self._cnt % (self._g_period) == 0:
            self.hooked_sess.run(self.g_min)
        self._cnt += 1


class MultiGPUGANTrainer(MultiGPUTrainerBase, FeedfreeTrainerBase):
    """
    A replacement of GANTrainer (optimize d and g one by one) with multi-gpu support.
    """
    def __init__(self, config):
        super(MultiGPUGANTrainer, self).__init__(config)
        self._nr_gpu = config.nr_tower
        assert self._nr_gpu > 1
        self._raw_devices = ['/gpu:{}'.format(k) for k in self.config.tower]
        self._input_source = StagingInputWrapper(QueueInput(config.dataflow), self._raw_devices)

    def _setup(self):
        super(MultiGPUGANTrainer, self)._setup()
        devices = [LeastLoadedDeviceSetter(d, self._raw_devices) for d in self._raw_devices]

        def get_cost():
            self.build_train_tower()
            return [self.model.d_loss, self.model.g_loss]
        cost_list = MultiGPUTrainerBase.build_on_multi_tower(
            self.config.tower, get_cost, devices)
        # simply average the cost. might be faster to average the gradients
        d_loss = tf.add_n([x[0] for x in cost_list]) * (1.0 / self._nr_gpu)
        g_loss = tf.add_n([x[1] for x in cost_list]) * (1.0 / self._nr_gpu)

        opt = self.model.get_optimizer()
        # run one d_min after one g_min
        self.g_min = opt.minimize(g_loss, var_list=self.model.g_vars,
                                  colocate_gradients_with_ops=True, name='g_op')
        with tf.control_dependencies([self.g_min]):
            self.d_min = opt.minimize(d_loss, var_list=self.model.d_vars,
                                      colocate_gradients_with_ops=True, name='d_op')
        self.train_op = self.d_min


class RandomZData(DataFlow):
    def __init__(self, shape):
        super(RandomZData, self).__init__()
        self.shape = shape

    def get_data(self):
        while True:
            yield [np.random.uniform(-1, 1, size=self.shape)]
