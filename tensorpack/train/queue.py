#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: queue.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import threading
import tensorflow as tf

from ..dataflow.common import RepeatedData
from ..tfutils.summary import summary_moving_average, add_moving_summary
from ..tfutils import get_global_step_var, TowerContext
from ..utils import logger
from ..callbacks.concurrency import StartProcOrThread
from ..tfutils.gradproc import apply_grad_processors

from .trainer import (FeedlessTrainer, MultiPredictorTowerTrainer,
        SingleCostFeedlessTrainer)

__all__ = ['QueueInputTrainerBase', 'QueueInputTrainer']

class EnqueueThread(threading.Thread):
    def __init__(self, trainer):
        super(EnqueueThread, self).__init__()
        self.name = 'EnqueueThread'
        self.daemon = True

        self.sess = trainer.sess
        self.coord = trainer.coord
        self.dataflow = RepeatedData(trainer.config.dataset, -1)
        self.input_vars = trainer.input_vars
        self.queue = trainer.input_queue

        self.op = self.queue.enqueue(self.input_vars)
        self.close_op = self.queue.close(cancel_pending_enqueues=True)
        self.size_op = self.queue.size()
        add_moving_summary(tf.cast(
            self.size_op, tf.float32, name='input_queue_size'))

    def run(self):
        self.dataflow.reset_state()
        with self.sess.as_default():
            try:
                while True:
                    for dp in self.dataflow.get_data():
                        if self.coord.should_stop():
                            return
                        feed = dict(zip(self.input_vars, dp))
                        #print 'TFQ:', self.sess.run([self.op, self.size_op], feed_dict=feed)[1]
                        self.op.run(feed_dict=feed)
            except tf.errors.CancelledError as e:
                pass
            except Exception:
                logger.exception("Exception in EnqueueThread:")
            finally:
                self.coord.request_stop()
                try:
                    self.sess.run(self.close_op)
                except RuntimeError:    # session already closed
                    pass
                logger.info("Enqueue Thread Exited.")

class QueueInputTrainerBase(FeedlessTrainer):
    def _build_enque_thread(self, input_queue=None):
        """ create a thread that keeps filling the queue """
        self.input_vars = self.model.get_input_vars()
        if input_queue is None:
            self.input_queue = tf.FIFOQueue(
                    50, [x.dtype for x in self.input_vars], name='input_queue')
        else:
            self.input_queue = input_queue
        input_th = EnqueueThread(self)
        self.config.callbacks.append(StartProcOrThread(input_th))

    def _get_input_tensors_noreuse(self):
        """ Dequeue a datapoint from input_queue and return.
            Can be called multiple times.
        """
        ret = self.input_queue.dequeue(name='input_deque')
        if isinstance(ret, tf.Tensor): # only one input
            ret = [ret]
        assert len(ret) == len(self.input_vars)
        for qv, v in zip(ret, self.input_vars):
            qv.set_shape(v.get_shape())

        # test the overhead of queue
        #with tf.device('/gpu:0'):
            #ret = [tf.Variable(tf.random_normal([128,224,224,3],
                #dtype=tf.float32), trainable=False),
                #tf.Variable(tf.ones([128], dtype=tf.int32), trainable=False)]
        return ret

class QueueInputTrainer(MultiPredictorTowerTrainer, QueueInputTrainerBase, SingleCostFeedlessTrainer):
    """ Single GPU Trainer, takes input from a queue"""

    def __init__(self, config, input_queue=None, predict_tower=None):
        """
        :param config: a `TrainConfig` instance
        :param input_queue: a `tf.QueueBase` instance to be used to buffer datapoints.
            Defaults to a FIFO queue of size 100.
        :param predict_tower: list of gpu relative idx to run prediction. default to be [0].
            Use -1 for cpu.
        """
        super(QueueInputTrainer, self).__init__(config)
        self._setup_predictor_factory(predict_tower)
        self._build_enque_thread(input_queue)

    def _setup(self):
        assert len(self.config.tower) == 1, \
                "QueueInputTrainer doesn't support multigpu! Use Sync/AsyncMultiGPUTrainer instead."
        with TowerContext(''):
            cost, grads = self._get_cost_and_grad()
        grads = apply_grad_processors(grads, self.model.get_gradient_processor())

        self.train_op = tf.group(
            self.config.optimizer.apply_gradients(grads, get_global_step_var()),
            summary_moving_average(), name='train_op')
        # skip training
        #self.train_op = tf.group(*self.dequed_inputs)

    def run_step(self):
        """ Simply run self.train_op"""
        self.sess.run(self.train_op)
        # debug-benchmark code:
        #run_metadata = tf.RunMetadata()
        #self.sess.run([self.train_op],
                #options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
                #run_metadata=run_metadata
                #)
        #from tensorflow.python.client import timeline
        #trace = timeline.Timeline(step_stats=run_metadata.step_stats)
        #trace_file = open('timeline.ctf.json', 'w')
        #trace_file.write(trace.generate_chrome_trace_format())
        #import sys; sys.exit()

