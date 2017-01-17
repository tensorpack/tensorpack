#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: GAN.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import tensorflow as tf
import numpy as np
import time
from tensorpack import (FeedfreeTrainerBase, TowerContext,
                        get_global_step_var, QueueInput, ModelDesc)
from tensorpack.tfutils.summary import summary_moving_average, add_moving_summary
from tensorpack.dataflow import DataFlow


class GANModelDesc(ModelDesc):

    def generate_code(self, factors=[], code_len=100, name='z'):
        """Generate input-node for noise data

        Examples:
            l1 = CategoricalDistribution("cat", 10) # len 10
            l2 = UniformDistribution("uni", 2)      # len 2

            z1 = l1.sample(BATCH, name='zc')
            z2 = l2.sample(BATCH, name='zu')

            z = self.generate_code([z1, z2], code_len=100)

        Args:
            factors (list, optional): tensors encoding different noise
            code_len (int, optional): total length of code
            name (str, optional): name for full code placeholder

        Returns:
            tf.Tensor: placeholder for noise
        """
        with tf.name_scope("latent_factors"):
            batch_size = tf.shape(factors[0])[0]

            factor_len = np.sum([z.get_shape().as_list()[1] for z in factors])
            noise_len = code_len - factor_len

            # generate fully noisy input
            z = tf.random_uniform(tf.stack([batch_size, noise_len]), -1, 1, name='z_train')
            # define placeholder
            z = tf.placeholder_with_default(z, [None, noise_len], name='z')

            factors.append(z)
            z = tf.concat_v2(factors, 1, name=name)
            return z

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


class GANTrainer(FeedfreeTrainerBase):
    def __init__(self, config):
        self._input_method = QueueInput(config.dataflow)
        super(GANTrainer, self).__init__(config)

    def _setup(self):
        super(GANTrainer, self)._setup()
        with TowerContext(''):
            actual_inputs = self._get_input_tensors()
            self.model.build_graph(actual_inputs)
        self.g_min = self.config.optimizer.minimize(self.model.g_loss,
                                                    var_list=self.model.g_vars, name='g_op')
        with tf.control_dependencies([self.g_min]):
            self.d_min = self.config.optimizer.minimize(self.model.d_loss,
                                                        var_list=self.model.d_vars, name='d_op')
        self.gs_incr = tf.assign_add(get_global_step_var(), 1, name='global_step_incr')
        self.summary_op = summary_moving_average()
        self.train_op = tf.group(self.d_min, self.summary_op, self.gs_incr)

    def run_step(self):
        self.sess.run(self.train_op)


class RandomZData(DataFlow):

    def __init__(self, shape):
        super(RandomZData, self).__init__()
        self.shape = shape

    def get_data(self):
        while True:
            yield [np.random.uniform(-1, 1, size=self.shape)]
