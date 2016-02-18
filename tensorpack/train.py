#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: train.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import tensorflow as tf
from itertools import count
import copy
import argparse
import re
import tqdm
from abc import ABCMeta

from .models import ModelDesc
from .dataflow.common import RepeatedData
from .utils import *
from .utils.concurrency import EnqueueThread
from .callbacks import *
from .utils.summary import summary_moving_average
from .utils.modelutils import describe_model
from .utils import logger
from .dataflow import DataFlow

class TrainConfig(object):
    """
    Config for training a model with a single loss
    """
    def __init__(self, **kwargs):
        """
        Args:
            dataset: the dataset to train. a tensorpack.dataflow.DataFlow instance.
            optimizer: a tf.train.Optimizer instance defining the optimizer for trainig.
            callbacks: a tensorpack.utils.callback.Callbacks instance. Define
                the callbacks to perform during training. has to contain a
                SummaryWriter and a PeriodicSaver
            session_config: a tf.ConfigProto instance to instantiate the
                session. default to a session running 1 GPU.
            session_init: a tensorpack.utils.sessinit.SessionInit instance to
                initialize variables of a session. default to a new session.
            model: a ModelDesc instance
            step_per_epoch: the number of steps (parameter updates) to perform
                in each epoch. default to dataset.size()
            max_epoch: maximum number of epoch to run training. default to 100
            nr_tower: int. number of towers. default to 1.
        """
        def assert_type(v, tp):
            assert isinstance(v, tp), v.__class__
        self.dataset = kwargs.pop('dataset')
        assert_type(self.dataset, DataFlow)
        self.optimizer = kwargs.pop('optimizer')
        assert_type(self.optimizer, tf.train.Optimizer)
        self.callbacks = kwargs.pop('callbacks')
        assert_type(self.callbacks, Callbacks)
        self.model = kwargs.pop('model')
        assert_type(self.model, ModelDesc)

        self.session_config = kwargs.pop('session_config', get_default_sess_config())
        assert_type(self.session_config, tf.ConfigProto)
        self.session_init = kwargs.pop('session_init', NewSession())
        assert_type(self.session_init, SessionInit)
        self.step_per_epoch = int(kwargs.pop('step_per_epoch', self.dataset.size()))
        self.max_epoch = int(kwargs.pop('max_epoch', 100))
        assert self.step_per_epoch > 0 and self.max_epoch > 0
        self.nr_tower = int(kwargs.pop('nr_tower', 1))
        assert len(kwargs) == 0, 'Unknown arguments: {}'.format(str(kwargs.keys()))

def summary_grads(grads):
    for grad, var in grads:
        if grad:
            # TODO also summary RMS and print
            tf.histogram_summary(var.op.name + '/gradients', grad)

def check_grads(grads):
    for grad, var in grads:
        assert grad is not None, "Grad is None for variable {}".format(var.name)
        tf.Assert(tf.reduce_all(tf.is_finite(var)), [var])

def scale_grads(grads, multiplier):
    ret = []
    for grad, var in grads:
        varname = var.name
        for regex, val in multiplier:
            if re.search(regex, varname):
                logger.info("Apply lr multiplier {} for {}".format(val, varname))
                ret.append((grad * val, var))
                break
        else:
            ret.append((grad, var))
    return ret

class Trainer(object):
    __metaclass__ = ABCMeta

    def __init__(self, config):
        """
        Config: a `TrainConfig` instance
        """
        assert isinstance(config, TrainConfig), type(config)
        self.config = config
        tf.add_to_collection(MODEL_KEY, config.model)

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def run_step(self):
        pass

    def main_loop(self):
        callbacks = self.config.callbacks
        with self.sess.as_default():
            try:
                logger.info("Start training with global_step={}".format(get_global_step()))
                callbacks.before_train()
                tf.get_default_graph().finalize()

                for epoch in xrange(1, self.config.max_epoch):
                    with timed_operation(
                        'Epoch {}, global_step={}'.format(
                            epoch, get_global_step() + self.config.step_per_epoch)):
                        for step in tqdm.trange(
                                self.config.step_per_epoch,
                            leave=True, mininterval=0.5,
                            dynamic_ncols=True, ascii=True):
                            if self.coord.should_stop():
                                return
                            self.run_step()
                            callbacks.trigger_step()
                        # note that summary_op will take a data from the queue
                        callbacks.trigger_epoch()
            except (KeyboardInterrupt, Exception):
                raise
            finally:
                self.coord.request_stop()
                # Do I need to run queue.close?
                callbacks.after_train()
                self.sess.close()

    def init_session_and_coord(self):
        self.sess = tf.Session(config=self.config.session_config)
        self.config.session_init.init(self.sess)

        # start training:
        self.coord = tf.train.Coordinator()
        tf.train.start_queue_runners(
            sess=self.sess, coord=self.coord, daemon=True, start=True)



class SimpleTrainer(Trainer):
    def run_step(self):
        try:
            data = next(self.data_producer)
        except StopIteration:
            self.data_producer = self.config.dataset.get_data()
            data = next(self.data_producer)
        feed = dict(zip(self.input_vars, data))
        self.sess.run([self.train_op], feed_dict=feed)    # faster since train_op return None

    def train(self):
        model = self.config.model
        input_vars = model.get_input_vars()
        self.input_vars = input_vars
        cost_var = model.get_cost(input_vars, is_training=True)
        avg_maintain_op = summary_moving_average(cost_var)

        grads = self.config.optimizer.compute_gradients(cost_var)
        check_grads(grads)
        grads = scale_grads(grads, model.get_lr_multiplier())
        summary_grads(grads)

        self.train_op = tf.group(
            self.config.optimizer.apply_gradients(grads, get_global_step_var()),
            avg_maintain_op)

        describe_model()
        self.init_session_and_coord()
        self.data_producer = self.config.dataset.get_data()
        self.main_loop()


class QueueInputTrainer(Trainer):
    """
    Trainer which builds a queue for input.
    Support multi GPU.
    """

    @staticmethod
    def _average_grads(tower_grads):
        ret = []
        for grad_and_vars in zip(*tower_grads):
            grad = tf.add_n([x[0] for x in grad_and_vars]) / float(len(tower_grads))
            v = grad_and_vars[0][1]
            ret.append((grad, v))
        return ret

    def train(self):
        model = self.config.model
        input_vars = model.get_input_vars()
        input_queue = model.get_input_queue()

        enqueue_op = input_queue.enqueue(input_vars)
        def get_model_inputs():
            model_inputs = input_queue.dequeue()
            if isinstance(model_inputs, tf.Tensor): # only one input
                model_inputs = [model_inputs]
            for qv, v in zip(model_inputs, input_vars):
                qv.set_shape(v.get_shape())
            return model_inputs

        # get gradients to update:
        if self.config.nr_tower > 1:
            logger.info("Training a model of {} tower".format(self.config.nr_tower))
            # to avoid repeated summary from each device
            coll_keys = [tf.GraphKeys.SUMMARIES, MOVING_SUMMARY_VARS_KEY]
            kept_summaries = {}
            grad_list = []
            for i in range(self.config.nr_tower):
                with tf.device('/gpu:{}'.format(i)), \
                        tf.name_scope('tower{}'.format(i)) as scope:
                    model_inputs = get_model_inputs()
                    cost_var = model.get_cost(model_inputs, is_training=True)
                    grad_list.append(
                        self.config.optimizer.compute_gradients(cost_var))

                    if i == 0:
                        tf.get_variable_scope().reuse_variables()
                        for k in coll_keys:
                            kept_summaries[k] = copy.copy(tf.get_collection(k))
            for k in coll_keys:
                del tf.get_collection(k)[:]
                tf.get_collection(k).extend(kept_summaries[k])
            grads = QueueInputTrainer._average_grads(grad_list)
        else:
            model_inputs = get_model_inputs()
            cost_var = model.get_cost(model_inputs, is_training=True)
            grads = self.config.optimizer.compute_gradients(cost_var)
        avg_maintain_op = summary_moving_average(cost_var)  # TODO(multigpu) average the cost from each device?

        check_grads(grads)
        grads = scale_grads(grads, model.get_lr_multiplier())
        summary_grads(grads)

        self.train_op = tf.group(
            self.config.optimizer.apply_gradients(grads, get_global_step_var()),
            avg_maintain_op)

        describe_model()
        self.init_session_and_coord()

        # create a thread that keeps filling the queue
        input_th = EnqueueThread(self.sess, self.coord, enqueue_op, self.config.dataset, input_queue)
        input_th.start()
        self.main_loop()

    def run_step(self):
        self.sess.run([self.train_op])    # faster since train_op return None


def start_train(config):
    #if config.model.get_input_queue() is not None:
        ## XXX get_input_queue is called twice
        #tr = QueueInputTrainer()
    #else:
        #tr = SimpleTrainer()
    #tr = SimpleTrainer(config)
    tr = QueueInputTrainer(config)
    tr.train()
