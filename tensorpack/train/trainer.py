# -*- coding: UTF-8 -*-
# File: trainer.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import tensorflow as tf
import threading
import time
import re
import functools
from six.moves import zip

from .base import Trainer
from ..dataflow.common import RepeatedData
from ..utils import *
from ..utils.concurrency import LoopThread
from ..tfutils.summary import summary_moving_average
from ..tfutils.modelutils import describe_model
from ..tfutils import *

__all__ = ['SimpleTrainer', 'QueueInputTrainer',
        'AsyncMultiGPUTrainer', 'SyncMultiGPUTrainer']

class SimpleTrainer(Trainer):
    def run_step(self):
        data = next(self.data_producer)
        feed = dict(zip(self.input_vars, data))
        self.sess.run([self.train_op], feed_dict=feed)    # faster since train_op return None

    def train(self):
        model = self.model
        self.input_vars = model.get_input_vars()
        model.build_graph(self.input_vars, True)
        cost_var = model.get_cost()
        tf.add_to_collection(MOVING_SUMMARY_VARS_KEY, cost_var)

        grads = self.config.optimizer.compute_gradients(cost_var)
        grads = self.process_grads(grads)

        avg_maintain_op = summary_moving_average()
        self.train_op = tf.group(
            self.config.optimizer.apply_gradients(grads, get_global_step_var()),
            avg_maintain_op)

        self.init_session_and_coord()
        describe_model()
        # create an infinte data producer
        self.data_producer = RepeatedData(self.config.dataset, -1).get_data()
        self.main_loop()

    def _trigger_epoch(self):
        if self.summary_op is not None:
            data = next(self.data_producer)
            feed = dict(zip(self.input_vars, data))
            summary_str = self.summary_op.eval(feed_dict=feed)
            self._process_summary(summary_str)

    def get_predict_func(self, input_names, output_names):
        input_vars = get_vars_by_names(input_names)
        for v in input_vars:
            assert v in self.input_vars
        output_vars = get_vars_by_names(output_names)
        def func(inputs):
            assert len(inputs) == len(input_vars)
            feed = dict(zip(input_vars, inputs))
            return self.sess.run(output_vars, feed_dict=feed)
        return func

class EnqueueThread(threading.Thread):
    def __init__(self, trainer, queue, enqueue_op, raw_input_var):
        super(EnqueueThread, self).__init__()
        self.sess = trainer.sess
        self.coord = trainer.coord
        self.dataflow = RepeatedData(trainer.config.dataset, -1)

        self.input_vars = raw_input_var
        self.op = enqueue_op
        self.queue = queue
        self.close_op = self.queue.close(cancel_pending_enqueues=True)

        self.size_op = self.queue.size()
        self.daemon = True

    def run(self):
        with self.sess.as_default():
            try:
                while True:
                    for dp in self.dataflow.get_data():
                        if self.coord.should_stop():
                            return
                        feed = dict(zip(self.input_vars, dp))
                        #_, size = self.sess.run([self.op, self.size_op], feed_dict=feed)
                        #print size
                        self.op.run(feed_dict=feed)
            except tf.errors.CancelledError as e:
                pass
            except Exception:
                logger.exception("Exception in EnqueueThread:")
                self.sess.run(self.close_op)
                self.coord.request_stop()
            finally:
                logger.info("Enqueue Thread Exited.")

class QueueInputTrainer(Trainer):
    """ Single GPU Trainer, takes input from a queue"""

    SUMMARY_BACKUP_KEYS = [tf.GraphKeys.SUMMARIES, MOVING_SUMMARY_VARS_KEY]

    def __init__(self, config, input_queue=None, predict_tower=None):
        """
        :param config: a `TrainConfig` instance
        :param input_queue: a `tf.QueueBase` instance to be used to buffer datapoints.
            Defaults to a FIFO queue of size 100.
        """
        super(QueueInputTrainer, self).__init__(config)
        self.input_vars = self.model.get_input_vars()
        if input_queue is None:
            self.input_queue = tf.FIFOQueue(
                    100, [x.dtype for x in self.input_vars], name='input_queue')
        else:
            self.input_queue = input_queue
        if predict_tower is None:
            # by default, use first training tower for prediction
            predict_tower = [0]
        self.predict_tower = predict_tower
        self.dequed_inputs = None

    def _get_model_inputs(self):
        """ Dequeue a datapoint from input_queue and return"""
        ret = self.input_queue.dequeue(name='input_deque')
        if isinstance(ret, tf.Tensor): # only one input
            ret = [ret]
        assert len(ret) == len(self.input_vars)
        for qv, v in zip(ret, self.input_vars):
            qv.set_shape(v.get_shape())
        return ret

    def _build_predict_tower(self):
        inputs = self.model.get_input_vars()
        tf.get_variable_scope().reuse_variables()
        for k in self.predict_tower:
            logger.info("Building graph for predict towerp{}...".format(k))
            with tf.device('/gpu:{}'.format(k)), \
                    tf.name_scope('towerp{}'.format(k)):
                self.model.build_graph(inputs, False)

    def _single_tower_grad(self):
        """ Get grad and cost for single-tower case"""
        self.dequed_inputs = model_inputs = self._get_model_inputs()
        self.model.build_graph(model_inputs, True)
        cost_var = self.model.get_cost()
        grads = self.config.optimizer.compute_gradients(cost_var)
        tf.add_to_collection(MOVING_SUMMARY_VARS_KEY, cost_var)
        return grads

    def _build_enque_thread(self):
        # create a thread that keeps filling the queue
        enqueue_op = self.input_queue.enqueue(self.input_vars)
        self.input_th = EnqueueThread(self, self.input_queue, enqueue_op, self.input_vars)
        self.extra_threads_procs.append(self.input_th)

    def train(self):
        assert self.config.nr_tower == 1, "QueueInputTrainer only supports 1 tower!"
        self.init_session_and_coord()
        self._build_enque_thread()

        grads = self._single_tower_grad()
        grads = self.process_grads(grads)
        describe_model()

        with freeze_collection(self.SUMMARY_BACKUP_KEYS):
            self._build_predict_tower()

        self.train_op = tf.group(
            self.config.optimizer.apply_gradients(grads, get_global_step_var()),
            summary_moving_average())

        self.main_loop()

    def run_step(self):
        """ just run self.train_op"""
        self.sess.run([self.train_op])

    def _trigger_epoch(self):
        # need to run summary_op every epoch
        # note that summary_op will take a data from the queue
        if self.summary_op is not None:
            summary_str = self.summary_op.eval()
            self._process_summary(summary_str)

    def get_predict_func(self, input_names, output_names, tower=0):
        """
        :param tower: return the kth predict_func
        :returns: a predictor function
        """
        tower = self.predict_tower[tower % len(self.predict_tower)]
        raw_input_vars = get_vars_by_names(input_names)
        output_names = ['towerp{}/'.format(tower) + n for n in output_names]
        output_vars = get_vars_by_names(output_names)
        def func(inputs):
            assert len(inputs) == len(raw_input_vars)
            feed = dict(zip(raw_input_vars, inputs))
            return self.sess.run(output_vars, feed_dict=feed)
        return func

    def get_predict_funcs(self, input_names, output_names, n):
        """ return n predicts functions evenly on each predict_tower"""
        return [self.get_predict_func(input_names, output_names, k)
                for k in range(n)]

class MultiGPUTrainer(QueueInputTrainer):
    """ Base class for multi-gpu training"""
    def __init__(self, config, input_queue=None, predict_tower=None):
        super(MultiGPUTrainer, self).__init__(config, input_queue, predict_tower)
        assert config.nr_tower > 1
        self.dequed_inputs = []

    @staticmethod
    def _average_grads(tower_grads):
        ret = []
        for grad_and_vars in zip(*tower_grads):
            v = grad_and_vars[0][1]
            try:
                grad = tf.add_n([x[0] for x in grad_and_vars]) / float(len(tower_grads))
            except AssertionError:
                logger.error("Error while processing gradients of {}".format(v.name))
                raise
            ret.append((grad, v))
        return ret

    def _multi_tower_grads(self):
        logger.info("Training a model of {} tower".format(self.config.nr_tower))

        grad_list = []
        for i in range(self.config.nr_tower):
            with tf.device('/gpu:{}'.format(i)), \
                    tf.name_scope('tower{}'.format(i)) as scope:
                logger.info("Building graph for training tower {}...".format(i))
                model_inputs = self._get_model_inputs()    # each tower dequeue from input queue
                self.dequed_inputs.append(model_inputs)
                self.model.build_graph(model_inputs, True)
                cost_var = self.model.get_cost() # build tower

                # gate_gradienst=0 seems to be faster?
                grad_list.append(
                    self.config.optimizer.compute_gradients(cost_var, gate_gradients=0))

                if i == 0:
                    tf.add_to_collection(MOVING_SUMMARY_VARS_KEY, cost_var)
                    tf.get_variable_scope().reuse_variables()
                    # avoid repeated summary from each device
                    backup = backup_collection(self.SUMMARY_BACKUP_KEYS)
        restore_collection(backup)
        return grad_list

class SyncMultiGPUTrainer(MultiGPUTrainer):
    def train(self):
        self.init_session_and_coord()
        self._build_enque_thread()

        grad_list = self._multi_tower_grads()

        grads = MultiGPUTrainer._average_grads(grad_list)
        grads = self.process_grads(grads)

        self.train_op = tf.group(
            self.config.optimizer.apply_gradients(grads, get_global_step_var()),
            summary_moving_average())
        describe_model()

        self._build_predict_tower()

        # [debug]: do nothing in training
        #self.train_op = self.dequed_inputs[0][0] + self.dequed_inputs[1][0]
        self.main_loop()

class AsyncMultiGPUTrainer(MultiGPUTrainer):
    def train(self):
        self.init_session_and_coord()
        self._build_enque_thread()

        grad_list = self._multi_tower_grads()
        # pretend to average the grads, in order to make async and
        # sync have consistent effective learning rate
        def scale(grads):
            return [(grad / self.config.nr_tower, var) for grad, var in grads]
        grad_list = map(scale, grad_list)
        grad_list = [self.process_grads(g) for g in grad_list]
        grads = grad_list[0]  # use grad from the first tower for the main iteration

        self.train_op = tf.group(
            self.config.optimizer.apply_gradients(grads, get_global_step_var()),
            summary_moving_average())
        describe_model()

        # prepare train_op for the rest of the towers
        self.threads = []
        for k in range(1, self.config.nr_tower):
            train_op = self.config.optimizer.apply_gradients(grad_list[k])
            f = lambda op=train_op: self.sess.run([op]) # avoid late-binding
            th = LoopThread(f)
            th.pause()
            th.start()
            self.threads.append(th)
        self.async_running = False

        self._build_predict_tower()

        # [debug]: do nothing in training
        #self.train_op = self.dequed_inputs[0][0] + self.dequed_inputs[1][0]
        self.main_loop()

    def run_step(self):
        if not self.async_running:
            self.async_running = True
            for th in self.threads: # resume all threads
                th.resume()
        self.sess.run([self.train_op])    # faster since train_op return None

    def _trigger_epoch(self):
        self.async_running = False
        for th in self.threads:
            th.pause()
        if self.summary_op is not None:
            summary_str = self.summary_op.eval()
            self._process_summary(summary_str)
