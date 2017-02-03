#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: GAN.py
# Author: TensorPack contributors

import tensorflow as tf
import numpy as np
import time
import six
from tensorpack import (FeedfreeTrainerBase, TowerContext,
                        QueueInput, ModelDesc)
from tensorpack.tfutils.summary import add_moving_summary
from tensorpack.tfutils.gradproc import apply_grad_processors, CheckGradient
from tensorpack.dataflow import DataFlow
from termcolor import colored
from tensorpack.utils import logger
import tensorpack.tfutils.symbolic_functions as symbf


class GANModelDesc(ModelDesc):
    def collect_variables(self):
        """Extract variables by prefix
        """
        all_vars = tf.trainable_variables()
        self.g_vars = [v for v in all_vars if v.name.startswith('gen/')]
        self.d_vars = [v for v in all_vars if v.name.startswith('discrim/')]

    def build_losses(self, logits_real, logits_fake):
        """D and G play two-player minimax game with value function V(G,D)


          min_G max _D V(D, G) = IE_{x ~ p_data} [log D(x)] + IE_{z ~ p_fake} [log (1 - D(G(z)))]

        Note, we swap 0, 1 labels as suggested in "Improving GANs".

        Args:
            logits_real (tf.Tensor): discrim logits from real samples
            logits_fake (tf.Tensor): discrim logits from fake samples produced by generator

        Returns:
            tf.Tensor: Description
        """
        with tf.name_scope("GAN_loss"):
            score_real = tf.sigmoid(logits_real)
            score_fake = tf.sigmoid(logits_fake)
            tf.summary.histogram('score-real', score_real)
            tf.summary.histogram('score-fake', score_fake)

            with tf.name_scope("discrim"):
                d_loss_pos = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=logits_real, labels=tf.zeros_like(logits_real)), name='loss_real')
                d_loss_neg = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=logits_fake, labels=tf.ones_like(logits_fake)), name='loss_fake')

                d_pos_acc = tf.reduce_mean(tf.cast(score_real < 0.5, tf.float32), name='accuracy_real')
                d_neg_acc = tf.reduce_mean(tf.cast(score_fake > 0.5, tf.float32), name='accuracy_fake')

                self.d_accuracy = tf.add(.5 * d_pos_acc, .5 * d_neg_acc, name='accuracy')
                self.d_loss = tf.add(.5 * d_loss_pos, .5 * d_loss_neg, name='loss')

            with tf.name_scope("gen"):
                self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=logits_fake, labels=tf.zeros_like(logits_fake)), name='loss')
                self.g_accuracy = tf.reduce_mean(tf.cast(score_fake < 0.5, tf.float32), name='accuracy')

            add_moving_summary(self.g_loss, self.d_loss, self.d_accuracy, self.g_accuracy)

    def get_gradient_processor_g(self):
        return [CheckGradient()]

    def get_gradient_processor_d(self):
        return [CheckGradient()]


class WGANModelDesc(GANModelDesc):
    """Model for Wasserstein-GAN.

    see https://arxiv.org/abs/1701.07875
    """

    def __init__(self, clamp_value=0.01, critic_runs=5):
        super(WGANModelDesc, self).__init__()
        self.clamp_value = clamp_value
        self.critic_runs = critic_runs

    def get_clipping_weights(self):
        raise NotImplementedError

    def build_losses(self, logits_real, logits_fake):
        with tf.name_scope("wasserstein_loss"):
            score_real = tf.reduce_mean(logits_real)
            score_fake = tf.reduce_mean(logits_fake)
            tf.summary.histogram('score-real', score_real)
            tf.summary.histogram('score-fake', score_fake)

            with tf.name_scope("discrim"):
                self.d_loss = tf.subtract(score_fake, score_real, name="loss")

            with tf.name_scope("gen"):
                self.g_loss = tf.multiply(-1.0, score_fake, name="loss")
            add_moving_summary(self.g_loss, self.d_loss)


class GANTrainer(FeedfreeTrainerBase):
    def __init__(self, config):
        self._input_method = QueueInput(config.dataflow)
        super(GANTrainer, self).__init__(config)

    def min_op(self, loss, vars, processor=None, name="opt_op"):
        grads = self.config.optimizer.compute_gradients(loss, var_list=vars)
        if processor:
            grads = apply_grad_processors(grads, processor)
        return self.config.optimizer.apply_gradients(grads, name=name)

    def create_min_ops(self):
        # optimize G
        self.g_min = self.min_op(self.model.g_loss, self.model.g_vars,
                                 self.model.get_gradient_processor_g(), name='g_op')

        # optimize D
        with tf.control_dependencies([self.g_min]):
            self.d_min = self.min_op(self.model.d_loss, self.model.d_vars,
                                     self.model.get_gradient_processor_d(), name='d_op')

    def _setup(self):
        super(GANTrainer, self)._setup()
        with TowerContext(''):
            actual_inputs = self._get_input_tensors()
            self.model.build_graph(actual_inputs)

        self.create_min_ops()
        self.train_op = self.d_min

    def run_step(self):
        ret = self.sess.run([self.train_op] + self.get_extra_fetches())
        return ret[1:]


class WGANTrainer(GANTrainer):
    def __init__(self, config):
        self._input_method = QueueInput(config.dataflow)
        super(WGANTrainer, self).__init__(config)

    def _setup(self):
        super(WGANTrainer, self)._setup()

    def create_min_ops(self):
        # optimize D
        self.d_min = self.min_op(self.model.d_loss, self.model.d_vars,
                                 self.model.get_gradient_processor_d(), name='d_op')

        # optimize G
        self.g_min = self.min_op(self.model.g_loss, self.model.g_vars,
                                 self.model.get_gradient_processor_g(), name='g_op')

        # add weight clipping to discriminator (alias critic)
        c = self.model.clamp_value
        weights = self.model.get_clipping_weights()
        w_names = [v.name for v in weights]
        prefix = colored("Parameters for clipping within range(%g, %g): " % (-c, c), 'cyan')
        logger.info(prefix + '\n' + '\n'.join(w_names))

        with tf.control_dependencies([self.d_min]):
            self.clipping_op = tf.group(*[symbf.clip(weight, -c, c) for weight in weights])

        self.d_min = self.clipping_op

    def run_step(self):
        for _ in six.moves.range(self.model.critic_runs):
            self.sess.run([self.d_min])
        ret = self.sess.run([self.g_min] + self.get_extra_fetches())
        return ret[1:]


class RandomZData(DataFlow):
    def __init__(self, shape):
        super(RandomZData, self).__init__()
        self.shape = shape

    def get_data(self):
        while True:
            yield [np.random.uniform(-1, 1, size=self.shape)]
