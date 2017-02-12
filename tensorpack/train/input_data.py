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
from ..tfutils import get_op_tensor_name
from ..utils import logger
from ..callbacks.concurrency import StartProcOrThread

__all__ = ['InputData', 'FeedfreeInput',
           'QueueInput', 'BatchQueueInput',
           'TensorInput', 'DummyConstantInput']


@six.add_metaclass(ABCMeta)
class InputData(object):
    """ Base class for the abstract InputData. """
    pass


class FeedInput(InputData):
    """ Input by iterating over a DataFlow and feed datapoints. """
    def __init__(self, ds):
        """
        Args:
            ds (DataFlow): the input DataFlow.
        """
        assert isinstance(ds, DataFlow), ds
        self.ds = ds

    def size(self):
        return self.ds.size()

    def _setup(self, trainer):
        self.input_placehdrs = trainer.model.get_reused_placehdrs()
        rds = RepeatedData(self.ds, -1)
        rds.reset_state()
        self.data_producer = rds.get_data()

    def next_feed(self):
        data = next(self.data_producer)
        feed = dict(zip(self.input_placehdrs, data))
        self._last_feed = feed
        return feed

    def last_feed(self):
        return self._last_feed


class FeedfreeInput(InputData):
    """ Abstract base for input without feed,
    e.g. by queue or other operations. """

    def get_input_tensors(self):
        """
        Returns:
            list: A list of tensors corresponding to the inputs of the model.
        """
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
        try:
            self.dataflow.reset_state()
            with self.sess.as_default():
                while True:
                    for dp in self.dataflow.get_data():
                        if self.coord.should_stop():
                            return
                        feed = dict(zip(self.placehdrs, dp))
                        # print 'qsize:', self.sess.run([self.op, self.size_op], feed_dict=feed)[1]
                        self.op.run(feed_dict=feed)
        except tf.errors.CancelledError:
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
    """ Enqueue datapoints from a DataFlow to a TF queue.
        And the model receives dequeued tensors.
    """

    def __init__(self, ds, queue=None):
        """
        Args:
            ds(DataFlow): the input DataFlow.
            queue (tf.QueueBase): Defaults to a FIFO queue of size 50.
        """
        assert isinstance(ds, DataFlow), ds
        self.queue = queue
        self.ds = ds

    def size(self):
        return self.ds.size()

    def _setup(self, trainer):
        self.input_placehdrs = trainer.model.get_reused_placehdrs()
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
        if isinstance(ret, tf.Tensor):  # only one input
            ret = [ret]
        assert len(ret) == len(self.input_placehdrs)
        for qv, v in zip(ret, self.input_placehdrs):
            qv.set_shape(v.get_shape())
        return ret


class BatchQueueInput(FeedfreeInput):
    """ Enqueue datapoints from a DataFlow to a TF queue.
        And the model receives batches formed by concatenating
        dequeued tensors.
    """
    def __init__(self, ds, batch_size, queue=None):
        """
        Args:
            ds(DataFlow): the input DataFlow.
            batch_size(int): the batch size.
            queue (tf.QueueBase): Defaults to a FIFO queue of size 3000.
        """
        assert isinstance(ds, DataFlow), ds
        self.queue = queue
        self.ds = ds
        self.batch_size = int(batch_size)

    def size(self):
        return self.ds.size() // self.batch_size

    def _setup(self, trainer):
        self.input_placehdrs = trainer.model.get_reused_placehdrs()
        assert len(self.input_placehdrs) > 0, \
            "QueueInput can only be used with input placeholders!"

        # prepare placeholders without the first dimension
        placehdrs_nobatch = []
        for p in self.input_placehdrs:
            placehdrs_nobatch.append(tf.placeholder(
                dtype=p.dtype, shape=p.get_shape().as_list()[1:],
                name=get_op_tensor_name(p.name)[0] + '-nobatch'))

        # dequeue_many requires fully-defined shapes
        shape_err = "Use of BatchQueueInput requires inputs to have fully-defined "
        "shapes except for the batch dimension"
        shapes = []
        for p in placehdrs_nobatch:
            assert p.get_shape().is_fully_defined(), shape_err
            shapes.append(p.get_shape())

        if self.queue is None:
            self.queue = tf.FIFOQueue(
                3000, [x.dtype for x in self.input_placehdrs],
                shapes=shapes,
                name='input_queue')
        for shp in self.queue.shapes:
            assert shp.is_fully_defined(), shape_err

        self.thread = EnqueueThread(
            trainer, self.queue, self.ds, placehdrs_nobatch)
        trainer.config.callbacks.append(StartProcOrThread(self.thread))

    def _get_input_tensors(self):
        ret = self.queue.dequeue_many(self.batch_size, name='input_deque')
        if isinstance(ret, tf.Tensor):  # only one input
            ret = [ret]
        assert len(ret) == len(self.input_placehdrs)
        for qv, v in zip(ret, self.input_placehdrs):
            shp = v.get_shape().as_list()
            shp[0] = self.batch_size
            qv.set_shape(shp)
        return ret


class DummyConstantInput(FeedfreeInput):
    """ Input with some random tensor placed on GPU.
        Useful for debugging performance issues """

    def __init__(self, shapes):
        """
        Args:
            shapes (list[list]): a list of fully-sepcified shapes.
        """
        self.shapes = shapes
        logger.warn("Using dummy input for debug!")

    def _get_input_tensors(self):
        placehdrs = self.input_placehdrs
        assert len(self.shapes) == len(placehdrs)
        ret = []
        for idx, p in enumerate(placehdrs):
            ret.append(tf.get_variable(
                'dummy-' + p.op.name, shape=self.shapes[idx],
                dtype=p.dtype, trainable=False))
        return ret


class TensorInput(FeedfreeInput):
    """ Input from a list of tensors, e.g. a TF data reading pipeline. """

    def __init__(self, get_tensor_fn, size=None):
        """
        Args:
            get_tensor_fn: a function which returns a list of input tensors
                when called.
            size(int): size of this input. Use None to leave it undefined.
        """
        self.get_tensor_fn = get_tensor_fn
        if size is not None:
            size = int(size)
            assert size > 0
        self._size = size

    def size(self):
        if self._size is None:
            raise NotImplementedError("size of TensorInput is undefined!")
        return self._size

    def _setup(self, trainer):
        pass

    def _get_input_tensors(self):
        return self.get_tensor_fn()
