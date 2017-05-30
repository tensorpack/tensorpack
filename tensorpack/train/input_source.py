#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: input_source.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import tensorflow as tf
try:
    from tensorflow.python.ops.data_flow_ops import StagingArea
except ImportError:
    pass

from itertools import chain
from abc import ABCMeta, abstractmethod
import six
from six.moves import range, zip

from .utils import get_placeholders_by_names, get_tensors_inputs
from ..dataflow import DataFlow, RepeatedData
from ..tfutils.summary import add_moving_summary
from ..tfutils import get_op_tensor_name
from ..tfutils.tower import get_current_tower_context
from ..utils import logger
from ..utils.argtools import memoized
from ..utils.concurrency import ShareSessionThread
from ..callbacks.concurrency import StartProcOrThread
from ..callbacks.base import Callback

__all__ = ['InputSource',
           'FeedInput', 'DataParallelFeedInput',
           'FeedfreeInput',
           'QueueInput', 'BatchQueueInput',
           'ZMQInput', 'DummyConstantInput', 'TensorInput',
           'StagingInputWrapper', 'ReorderInputSource']


@six.add_metaclass(ABCMeta)
class InputSource(object):
    """ Base class for the abstract InputSource. """

    @abstractmethod
    def get_input_tensors(self):
        """
        Returns:
            list: A list of tensors corresponding to the inputs of the model,
                used as input of :func:`build_graph`.
                For non-placeholder tensors, should always create and return new tensors when called.
        """

    def setup(self, model):
        pass

    def setup_training(self, trainer):
        self.setup(trainer.model)

    @abstractmethod
    def reset_state(self):
        pass

    @abstractmethod
    def next_feed(self):
        """
        Returns:
            a feed_dict of {Tensor: data}, to be used to run the steps
        """
        pass


class FeedInput(InputSource):
    """ Input by iterating over a DataFlow and feed datapoints. """
    def __init__(self, ds, input_names=None):
        """
        Args:
            ds (DataFlow): the input DataFlow.
            input_names (list[str]): input names this DataFlow maps to
        """
        assert isinstance(ds, DataFlow), ds
        if input_names is not None:
            assert isinstance(input_names, (list, tuple)), input_names
        self.ds = ds
        self._input_names = input_names

    def size(self):
        return self.ds.size()

    def setup(self, model):
        self._all_placehdrs = model.get_reused_placehdrs()
        if self._input_names is None:
            self._placehdrs_to_feed = self._all_placehdrs
        else:
            self._placehdrs_to_feed = get_placeholders_by_names(
                self._all_placehdrs, self._input_names)

        self.reset_state()

    def reset_state(self):
        rds = RepeatedData(self.ds, -1)
        rds.reset_state()
        self.data_producer = rds.get_data()

    def get_input_tensors(self):
        return self._all_placehdrs

    def next_feed(self):
        dp = next(self.data_producer)
        return dict(zip(self._placehdrs_to_feed, dp))


class DataParallelFeedInput(FeedInput):
    """
    Input by feeding k datapoints to k copies of placeholders located on k towers.
    """
    def __init__(self, ds, tower_names, input_names=None):
        super(DataParallelFeedInput, self).__init__(ds, input_names)
        self._tower_names = tower_names
        self._nr_tower = len(tower_names)

    def setup(self, model):
        self._placehdrs_per_tower = []
        self._feed_placehdrs_per_tower = []
        for tname in self._tower_names:
            # build a list of placeholders for each tower
            self._placehdrs_per_tower.append(
                model.build_placeholders(
                    prefix=tname + '/'))

        # apply input mapping and store results in feed_placehdrs_per_tower
        if self._input_names is None:
            self._feed_placehdrs_per_tower = self._placehdrs_per_tower
        else:
            for phdrs, tname in zip(
                    self._placehdrs_per_tower, self._tower_names):
                input_names = [tname + '/' + n for n in self._input_names]
                # input_names to be used for this specific tower
                self._feed_placehdrs_per_tower.append(
                    get_placeholders_by_names(phdrs, input_names))
                print(self._feed_placehdrs_per_tower[-1])
        self.reset_state()

    def get_input_tensors(self):
        # return placeholders for each tower
        ctx = get_current_tower_context()
        return self._placehdrs_per_tower[ctx.index]

    def next_feed(self, cnt=None):
        """
        Args:
            cnt: how many towers to feed to. Defaults to the total number of towers
        """
        if cnt is None:
            cnt = self._nr_tower
        feed = {}
        for t in range(cnt):
            dp = next(self.data_producer)
            f = dict(zip(self._feed_placehdrs_per_tower[t], dp))
            feed.update(f)
        return feed


class FeedfreeInput(InputSource):
    """ Abstract base for input without feed,
    e.g. by queue or other operations. """

    def reset_state(self):
        # TODO cannot reset
        pass

    def next_feed(self):
        return {}


# TODO enqueu_many? https://github.com/tensorflow/tensorflow/issues/7817#issuecomment-282053155
class EnqueueThread(ShareSessionThread):
    def __init__(self, queue, ds, placehdrs):
        super(EnqueueThread, self).__init__()
        self.name = 'EnqueueThread'
        self.daemon = True

        self.dataflow = ds
        self.queue = queue

        self.placehdrs = placehdrs

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
            queue (tf.QueueBase): A :class:`tf.QueueBase` whose type
                should match the corresponding InputDesc of the model.
                Defaults to a FIFO queue of size 50.
        """
        assert isinstance(ds, DataFlow), ds
        self.queue = queue
        self.ds = ds

    def size(self):
        return self.ds.size()

    # TODO use input data mapping. not all placeholders are needed
    def setup(self, model):
        logger.info("Setting up the queue for CPU prefetching ...")
        self.input_placehdrs = model.get_reused_placehdrs()
        assert len(self.input_placehdrs) > 0, \
            "QueueInput has to be used with some InputDesc!"
        if self.queue is None:
            self.queue = tf.FIFOQueue(
                50, [x.dtype for x in self.input_placehdrs],
                name='input_queue')
        self.thread = EnqueueThread(self.queue, self.ds, self.input_placehdrs)

    def setup_training(self, trainer):
        super(QueueInput, self).setup_training(trainer)
        cb = StartProcOrThread(self.thread)
        cb._chief_only = False
        trainer.register_callback(cb)

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
            queue (tf.QueueBase): A :class:`tf.QueueBase` whose type
                should match the corresponding InputDesc of the model.
                Defaults to a FIFO queue of size 3000.
        """
        assert isinstance(ds, DataFlow), ds
        self.queue = queue
        self.ds = ds
        self.batch_size = int(batch_size)

    def size(self):
        return self.ds.size() // self.batch_size

    def setup(self, model):
        logger.info("Setting up the queue for CPU prefetching ...")
        self.input_placehdrs = model.get_reused_placehdrs()
        assert len(self.input_placehdrs) > 0, \
            "BatchQueueInput has to be used with some InputDesc!"

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


class TensorInput(FeedfreeInput):
    """ Input from a list of tensors, e.g. a TF data reading pipeline. """

    def __init__(self, get_tensor_fn, size=None):
        """
        Args:
            get_tensor_fn: a function which returns a list of input tensors
                when called. It will be called under a TowerContext.
            size(int): size of this input. Use None to leave it undefined.
            input_names (list[str]): input names the tensors maps to. Defaults
                to be all the inputs of the model.
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


class DummyConstantInput(TensorInput):
    """ Input with some random tensor placed on GPU.
        Useful for debugging performance issues """
    def __init__(self, shapes):
        """
        Args:
            shapes (list[list]): a list of fully-sepcified shapes.
        """
        self.shapes = shapes
        logger.warn("Using dummy input for debug!")

        def fn():
            tlist = []
            ctx = get_current_tower_context()
            assert len(self.shapes) == len(self.input_placehdrs)
            for idx, p in enumerate(self.input_placehdrs):
                tlist.append(tf.get_variable(
                    'dummy-{}-{}'.format(p.op.name, ctx.index), shape=self.shapes[idx],
                    dtype=p.dtype, trainable=False))
            return tlist
        super(DummyConstantInput, self).__init__(fn)

    def setup(self, model):
        self.input_placehdrs = model.get_reused_placehdrs()


# TODO doesn't support remapping
class ZMQInput(TensorInput):
    def __init__(self, endpoint):
        self._endpoint = endpoint

        from tensorpack.user_ops import zmq_recv

        def fn():
            ret = zmq_recv(self._endpoint, [x.dtype for x in self.input_placehdrs])
            if isinstance(ret, tf.Tensor):
                ret = [ret]
            assert len(ret) == len(self.input_placehdrs)
            for qv, v in zip(ret, self.input_placehdrs):
                qv.set_shape(v.get_shape())
            return ret
        super(ZMQInput, self).__init__(fn)

    def setup(self, model):
        self.input_placehdrs = model.get_reused_placehdrs()
        assert len(self.input_placehdrs) > 0, \
            "ZMQInput has to be used with InputDesc!"


class StagingInputWrapper(FeedfreeInput):
    """
    A wrapper around a feedfree input, to prefetch it in StagingArea (usually on GPUs).
    """
    class StagingCallback(Callback):
        """
        A callback registered by this input source, to make sure stage/unstage
        is run at each step.
        """
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
        """
        Args:
            input: a :class:`FeedfreeInput`
            devices: list of devices to be used for each training tower
            nr_stage: number of elements to prefetch
        """
        assert isinstance(input, FeedfreeInput), input
        self._input = input
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
        logger.info("Setting up the StageAreas for GPU prefetching ...")
        for idx, device in enumerate(self._devices):
            with tf.device(device):
                inputs = self._input.get_input_tensors()
                dtypes = [x.dtype for x in inputs]
                stage = StagingArea(dtypes, shapes=None)
                self._stage_ops.append(stage.put(inputs))
                self._areas.append(stage)
                outputs = stage.get()
                if isinstance(outputs, tf.Tensor):  # when size=1, TF doesn't return a list
                    outputs = [outputs]
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


class ReorderInputSource(FeedfreeInput):
    """
    When an InputSource only maps to a subset of the InputDesc of the model,
    wrap it with :class:`ReorderInputSource`.
    """
    def __init__(self, input, names):
        """
        Args:
            input(TensorInput): a TensorInput, whose tensors will get mapped.
            names(list[str]): list of input names corresponding to the tensors
                produced by ``input``.
        """
        assert isinstance(input, TensorInput), input
        self._input = input
        assert isinstance(names, (list, tuple)), names
        self._names = names

    def size(self):
        return self._input.size()

    def setup(self, model):
        self._all_placehdrs = model.get_reused_placehdrs()
        self._input.setup(model)

    def setup_training(self, trainer):
        self._all_placehdrs = trainer.model.get_reused_placehdrs()
        self._input.setup_training(trainer)

    def reset_state(self):
        self._input.reset_state()

    def get_input_tensors(self):
        ret = self._input.get_input_tensors()
        return get_tensors_inputs(
            self._all_placehdrs, ret, self._names)
