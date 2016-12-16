#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: input_data.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import tensorflow as tf
import threading
from abc import ABCMeta, abstractmethod
import six

from ..dataflow import DataFlow, RepeatedData
from ..tfutils.summary import add_moving_summary
from ..utils import logger
from ..callbacks.concurrency import StartProcOrThread

__all__ = ['QueueInput', 'FeedfreeInput', 'TensorInput',
        'DummyConstantInput']

@six.add_metaclass(ABCMeta)
class InputData(object):
    pass

class FeedInput(InputData):
    def __init__(self, ds):
        assert isinstance(ds, DataFlow), ds
        self.ds = ds

    def size(self):
        return self.ds.size()

    def _setup(self, trainer):
        self.input_vars = trainer.model.get_input_vars()
        rds = RepeatedData(self.ds, -1)
        rds.reset_state()
        self.data_producer = rds.get_data()

    def next_feed(self):
        data = next(self.data_producer)
        feed = dict(zip(self.input_vars, data))
        return feed

class FeedfreeInput(InputData):
    def get_input_tensors(self):
        return self._get_input_tensors()

    @abstractmethod
    def _get_input_tensors(self):
        """
        always create and return a list of new input tensors
        """

class EnqueueThread(threading.Thread):
    def __init__(self, trainer, queue, ds, input_placehdrs):
        super(EnqueueThread, self).__init__()
        self.name = 'EnqueueThread'
        self.daemon = True

        self.dataflow = ds
        self.queue = queue

        self.sess = trainer.sess
        self.coord = trainer.coord
        self.placehdrs = input_placehdrs

        self.op = self.queue.enqueue(self.placehdrs)
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
                        feed = dict(zip(self.placehdrs, dp))
                        #print 'qsize:', self.sess.run([self.op, self.size_op], feed_dict=feed)[1]
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

class QueueInput(FeedfreeInput):
    def __init__(self, ds, queue=None):
        """
        :param ds: a `DataFlow` instance
        :param queue: a `tf.QueueBase` instance to be used to buffer datapoints.
            Defaults to a FIFO queue of size 50.
        """
        assert isinstance(ds, DataFlow), ds
        self.queue = queue
        self.ds = ds

    def size(self):
        return self.ds.size()

    def _setup(self, trainer):
        self.input_placehdrs = trainer.model.get_input_vars()
        assert len(self.input_placehdrs) > 0, \
                "QueueInput can only be used with input placeholders!"
        if self.queue is None:
            self.queue = tf.FIFOQueue(
                    50, [x.dtype for x in self.input_placehdrs],
                    name='input_queue')
        self.thread = EnqueueThread(
                trainer, self.queue, self.ds, self.input_placehdrs)
        trainer.config.callbacks.append(StartProcOrThread(self.thread))

    def _get_input_tensors(self):
        ret = self.queue.dequeue(name='input_deque')
        if isinstance(ret, tf.Tensor): # only one input
            ret = [ret]
        assert len(ret) == len(self.input_placehdrs)
        for qv, v in zip(ret, self.input_placehdrs):
            qv.set_shape(v.get_shape())

        # test the overhead of queue
        #with tf.device('/gpu:0'):
            #ret = [tf.Variable(tf.random_normal([128,224,224,3],
                #dtype=tf.float32), trainable=False),
                #tf.Variable(tf.ones([128], dtype=tf.int32), trainable=False)]
        return ret

class DummyConstantInput(QueueInput):
    """ only for debugging performance issues """
    def __init__(self, ds, shapes):
        super(DummyConstantInput, self).__init__(ds)
        self.shapes = shapes
        logger.warn("Using dummy input for debug!")

    def _get_input_tensors(self):
        placehdrs = self.input_placehdrs
        assert len(self.shapes) == len(placehdrs)
        ret = []
        for idx, p in enumerate(placehdrs):
            with tf.device('/gpu:0'):
                ret.append(tf.get_variable('dummy-' + p.op.name,
                    shape=self.shapes[idx], dtype=p.dtype, trainable=False,
                    initializer=tf.constant_initializer()))
        return ret

class TensorInput(FeedfreeInput):
    def __init__(self, get_tensor_fn, size=None):
        self.get_tensor_fn = get_tensor_fn
        self._size = size

    def size(self):
        if self._size is None:
            raise ValueError("size of TensorInput is undefined!")
        return self._size

    def _setup(self, trainer):
        pass

    def _get_input_tensors(self):
        return self.get_tensor_fn()
