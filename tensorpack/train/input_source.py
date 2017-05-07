#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: input_source.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import tensorflow as tf
try:
    from tensorflow.contrib.staging import StagingArea
except ImportError:
    pass

from itertools import chain
from abc import ABCMeta, abstractmethod
import six
from six.moves import range

from ..dataflow import DataFlow, RepeatedData
from ..tfutils.summary import add_moving_summary
from ..tfutils import get_op_tensor_name
from ..tfutils.tower import get_current_tower_context
from ..utils import logger
from ..utils.argtools import memoized
from ..utils.concurrency import ShareSessionThread
from ..callbacks.concurrency import StartProcOrThread
from ..callbacks.base import Callback

__all__ = ['InputSource', 'FeedfreeInput',
           'QueueInput', 'BatchQueueInput',
           'ZMQInput',
           'DummyConstantInput', 'TensorInput', 'StagingInputWrapper']


@six.add_metaclass(ABCMeta)
class InputSource(object):
    """ Base class for the abstract InputSource. """

    @abstractmethod
    def get_input_tensors(self):
        """
        Returns:
            list: A list of tensors corresponding to the inputs of the model.
                Always create and return a list of new input tensors when called.
        """

    def setup(self, model):
        pass

    def setup_training(self, trainer):
        self.setup(trainer.model)

    @abstractmethod
    def reset_state(self):
        pass

    def next_feed(self):
        return []


class FeedInput(InputSource):
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

    def setup(self, model):
        self.input_placehdrs = model.get_reused_placehdrs()
        rds = RepeatedData(self.ds, -1)
        rds.reset_state()
        self.data_producer = rds.get_data()

    def reset_state(self):
        rds = RepeatedData(self.ds, -1)
        rds.reset_state()
        self.data_producer = rds.get_data()

    def get_input_tensors(self):
        return self.input_placehdrs

    def next_feed(self):
        return next(self.data_producer)


class FeedfreeInput(InputSource):
    """ Abstract base for input without feed,
    e.g. by queue or other operations. """

    def reset_state(self):
        # TODO cannot reset
        pass


# TODO enqueu_many? https://github.com/tensorflow/tensorflow/issues/7817#issuecomment-282053155
class EnqueueThread(ShareSessionThread):
    def __init__(self, queue, ds, input_placehdrs):
        super(EnqueueThread, self).__init__()
        self.name = 'EnqueueThread'
        self.daemon = True

        self.dataflow = ds
        self.queue = queue

        self.placehdrs = input_placehdrs

        self.op = self.queue.enqueue(self.placehdrs)
        self.close_op = self.queue.close(cancel_pending_enqueues=True)
        self.size_op = self.queue.size()
        add_moving_summary(tf.cast(
            self.size_op, tf.float32, name='input_queue_size'))

    def run(self):
        with self.default_sess():
            try:
                self.dataflow.reset_state()
                while True:
                    for dp in self.dataflow.get_data():
                        feed = dict(zip(self.placehdrs, dp))
                        # print 'qsize:', self.sess.run([self.op, self.size_op], feed_dict=feed)[1]
                        self.op.run(feed_dict=feed)
            except (tf.errors.CancelledError, tf.errors.OutOfRangeError):
                pass
            except Exception:
                logger.exception("Exception in EnqueueThread:")
            finally:
                try:
                    self.close_op.run()
                except Exception:
                    pass
                logger.info("EnqueueThread Exited.")


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

    # TODO XXX use input data mapping. not all placeholders are needed
    def setup(self, model):
        self.input_placehdrs = model.get_reused_placehdrs()
        assert len(self.input_placehdrs) > 0, \
            "QueueInput has to be used with input placeholders!"
        if self.queue is None:
            self.queue = tf.FIFOQueue(
                50, [x.dtype for x in self.input_placehdrs],
                name='input_queue')
        self.thread = EnqueueThread(self.queue, self.ds, self.input_placehdrs)

    def setup_training(self, trainer):
        super(QueueInput, self).setup_training(trainer)
        trainer.register_callback(StartProcOrThread(self.thread))

    def get_input_tensors(self):
        with tf.device('/cpu:0'):
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

    def setup(self, model):
        self.input_placehdrs = model.get_reused_placehdrs()
        assert len(self.input_placehdrs) > 0, \
            "BatchQueueInput has to be used with input placeholders!"

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

        self.thread = EnqueueThread(self.queue, self.ds, placehdrs_nobatch)

    def setup_training(self, trainer):
        super(BatchQueueInput, self).setup_training(trainer)
        trainer.register_callback(StartProcOrThread(self.thread))

    def get_input_tensors(self):
        with tf.device('/cpu:0'):
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

    def setup(self, model):
        self.input_placehdrs = model.get_reused_placehdrs()

    def setup_training(self, trainer):
        super(DummyConstantInput, self).setup_training(trainer)

        nr_tower = trainer.config.nr_tower

        placehdrs = self.input_placehdrs
        assert len(self.shapes) == len(placehdrs)
        self.tensors = []

        # don't share variables
        for tower in range(nr_tower):
            tlist = []
            with tf.device('/gpu:{}'.format(tower)):
                for idx, p in enumerate(placehdrs):
                    tlist.append(tf.get_variable(
                        'dummy-{}-{}'.format(p.op.name, tower), shape=self.shapes[idx],
                        dtype=p.dtype, trainable=False))
            self.tensors.append(tlist)

    def get_input_tensors(self):
        ctx = get_current_tower_context()
        ret = self.tensors[ctx.index]
        return ret


class TensorInput(FeedfreeInput):
    """ Input from a list of tensors, e.g. a TF data reading pipeline. """

    def __init__(self, get_tensor_fn, size=None):
        """
        Args:
            get_tensor_fn: a function which returns a list of input tensors
                when called. It will be called under a TowerContext.
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

    def get_input_tensors(self):
        return self.get_tensor_fn()


class ZMQInput(FeedfreeInput):
    def __init__(self, endpoint):
        self._endpoint = endpoint

    def size(self):
        raise NotImplementedError()

    def setup(self, model):
        self.input_placehdrs = model.get_reused_placehdrs()
        assert len(self.input_placehdrs) > 0, \
            "ZMQInput has to be used with input placeholders!"

    def get_input_tensors(self):
        from tensorpack.user_ops import zmq_recv
        ret = zmq_recv(self._endpoint, [x.dtype for x in self.input_placehdrs])
        if isinstance(ret, tf.Tensor):
            ret = [ret]
        assert len(ret) == len(self.input_placehdrs)
        for qv, v in zip(ret, self.input_placehdrs):
            qv.set_shape(v.get_shape())
        return ret


class StagingInputWrapper(FeedfreeInput):
    class StagingCallback(Callback):
        def __init__(self, stage_op, unstage_op, nr_stage):
            self.nr_stage = nr_stage
            self.stage_op = stage_op
            self.fetches = tf.train.SessionRunArgs(
                fetches=[stage_op, unstage_op])

        def _before_train(self):
            # pre-fill the staging area
            for k in range(self.nr_stage):
                self.stage_op.run()

        def _before_run(self, ctx):
            return self.fetches

    def __init__(self, input, devices, nr_stage=5):
        self._input = input
        assert isinstance(input, FeedfreeInput)
        self._devices = devices
        self._nr_stage = nr_stage
        self._areas = []
        self._stage_ops = []
        self._unstage_ops = []

    def setup(self, model):
        self._input.setup(model)
        self.setup_staging_areas()

    def setup_training(self, trainer):
        self._input.setup_training(trainer)
        self.setup_staging_areas()

        trainer.register_callback(
            StagingInputWrapper.StagingCallback(
                self.get_stage_op(), self.get_unstage_op(), self._nr_stage))

    def setup_staging_areas(self):
        for idx, device in enumerate(self._devices):
            with tf.device(device):
                inputs = self._input.get_input_tensors()
                dtypes = [x.dtype for x in inputs]
                stage = StagingArea(dtypes, shapes=None)
                self._stage_ops.append(stage.put(inputs))
                self._areas.append(stage)
                outputs = stage.get()
                for vin, vout in zip(inputs, outputs):
                    vout.set_shape(vin.get_shape())
                self._unstage_ops.append(outputs)

    def size(self):
        return self._input.size()

    def get_input_tensors(self):
        ctx = get_current_tower_context()
        ret = self._unstage_ops[ctx.index]
        return ret

    @staticmethod
    def get_staging_name(idx):
        return 'StagingArea{}'.format(idx)

    @memoized
    def get_stage_op(self):
        return tf.group(*self._stage_ops)

    @memoized
    def get_unstage_op(self):
        all_outputs = list(chain.from_iterable(self._unstage_ops))
        return tf.group(*all_outputs)
