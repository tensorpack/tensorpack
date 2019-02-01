# -*- coding: utf-8 -*-
# File: GAN.py
# Author: Yuxin Wu

import numpy as np
import tensorflow as tf

from tensorpack import BatchNorm, DataFlow, ModelDescBase, StagingInput, TowerTrainer, argscope
from tensorpack.graph_builder import DataParallelBuilder, LeastLoadedDeviceSetter
from tensorpack.tfutils.summary import add_moving_summary
from tensorpack.tfutils.tower import TowerContext, TowerFuncWrapper
from tensorpack.utils import logger
from tensorpack.utils.argtools import memoized_method
from tensorpack.utils.develop import deprecated


class GANModelDesc(ModelDescBase):
    def collect_variables(self, g_scope='gen', d_scope='discrim'):
        """
        Assign `self.g_vars` to the parameters under scope `g_scope`,
        and same with `self.d_vars`.
        """
        self.g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, g_scope)
        assert self.g_vars
        self.d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, d_scope)
        assert self.d_vars

    def build_losses(self, logits_real, logits_fake):
        """
        Build standard GAN loss and set `self.g_loss` and `self.d_loss`.

        D and G play two-player minimax game with value function V(G,D)

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

    def build_graph(self, *inputs):
        """
        Have to build one tower and set the following attributes:
        g_loss, d_loss, g_vars, d_vars.
        """
        pass

    @memoized_method
    def get_optimizer(self):
        return self.optimizer()


class GANTrainer(TowerTrainer):

    def __init__(self, input, model, num_gpu=1):
        """
        Args:
            input (InputSource):
            model (GANModelDesc):
        """
        super(GANTrainer, self).__init__()
        assert isinstance(model, GANModelDesc), model

        if num_gpu > 1:
            input = StagingInput(input)

        # Setup input
        cbs = input.setup(model.get_inputs_desc())
        self.register_callback(cbs)

        if num_gpu <= 1:
            self._build_gan_trainer(input, model)
        else:
            self._build_multigpu_gan_trainer(input, model, num_gpu)

    def _build_gan_trainer(self, input, model):
        """
        We need to set tower_func because it's a TowerTrainer,
        and only TowerTrainer supports automatic graph creation for inference during training.

        If we don't care about inference during training, using tower_func is
        not needed. Just calling model.build_graph directly is OK.
        """
        # Build the graph
        self.tower_func = TowerFuncWrapper(model.build_graph, model.get_inputs_desc())
        with TowerContext('', is_training=True):
            self.tower_func(*input.get_input_tensors())
        opt = model.get_optimizer()

        # Define the training iteration
        # by default, run one d_min after one g_min
        with tf.name_scope('optimize'):
            g_min = opt.minimize(model.g_loss, var_list=model.g_vars, name='g_op')
            with tf.control_dependencies([g_min]):
                d_min = opt.minimize(model.d_loss, var_list=model.d_vars, name='d_op')
        self.train_op = d_min

    def _build_multigpu_gan_trainer(self, input, model, num_gpu):
        assert num_gpu > 1
        raw_devices = ['/gpu:{}'.format(k) for k in range(num_gpu)]

        # Build the graph with multi-gpu replication
        def get_cost(*inputs):
            model.build_graph(*inputs)
            return [model.d_loss, model.g_loss]

        self.tower_func = TowerFuncWrapper(get_cost, model.get_inputs_desc())
        devices = [LeastLoadedDeviceSetter(d, raw_devices) for d in raw_devices]
        cost_list = DataParallelBuilder.build_on_towers(
            list(range(num_gpu)),
            lambda: self.tower_func(*input.get_input_tensors()),
            devices)
        # For simplicity, average the cost here. It might be faster to average the gradients
        with tf.name_scope('optimize'):
            d_loss = tf.add_n([x[0] for x in cost_list]) * (1.0 / num_gpu)
            g_loss = tf.add_n([x[1] for x in cost_list]) * (1.0 / num_gpu)

            opt = model.get_optimizer()
            # run one d_min after one g_min
            g_min = opt.minimize(g_loss, var_list=model.g_vars,
                                 colocate_gradients_with_ops=True, name='g_op')
            with tf.control_dependencies([g_min]):
                d_min = opt.minimize(d_loss, var_list=model.d_vars,
                                     colocate_gradients_with_ops=True, name='d_op')
        # Define the training iteration
        self.train_op = d_min


class MultiGPUGANTrainer(GANTrainer):
    """
    A replacement of GANTrainer (optimize d and g one by one) with multi-gpu support.
    """

    @deprecated("Please use GANTrainer and set num_gpu", "2019-01-31")
    def __init__(self, num_gpu, input, model):
        super(MultiGPUGANTrainer, self).__init__(input, model, 1)


class SeparateGANTrainer(TowerTrainer):
    """ A GAN trainer which runs two optimization ops with a certain ratio."""
    def __init__(self, input, model, d_period=1, g_period=1):
        """
        Args:
            d_period(int): period of each d_opt run
            g_period(int): period of each g_opt run
        """
        super(SeparateGANTrainer, self).__init__()
        self._d_period = int(d_period)
        self._g_period = int(g_period)
        assert min(d_period, g_period) == 1

        # Setup input
        cbs = input.setup(model.get_inputs_desc())
        self.register_callback(cbs)

        # Build the graph
        self.tower_func = TowerFuncWrapper(model.build_graph, model.get_inputs_desc())
        with TowerContext('', is_training=True), \
                argscope(BatchNorm, internal_update=True):
            # should not hook the updates to both train_op, it will hurt training speed.
            self.tower_func(*input.get_input_tensors())
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        if len(update_ops):
            logger.warn("Found {} ops in UPDATE_OPS collection!".format(len(update_ops)))
            logger.warn("Using SeparateGANTrainer with UPDATE_OPS may hurt your training speed a lot!")

        opt = model.get_optimizer()
        with tf.name_scope('optimize'):
            self.d_min = opt.minimize(
                model.d_loss, var_list=model.d_vars, name='d_min')
            self.g_min = opt.minimize(
                model.g_loss, var_list=model.g_vars, name='g_min')

    def run_step(self):
        # Define the training iteration
        if self.global_step % (self._d_period) == 0:
            self.hooked_sess.run(self.d_min)
        if self.global_step % (self._g_period) == 0:
            self.hooked_sess.run(self.g_min)


class RandomZData(DataFlow):
    def __init__(self, shape):
        super(RandomZData, self).__init__()
        self.shape = shape

    def __iter__(self):
        while True:
            yield [np.random.uniform(-1, 1, size=self.shape)]
