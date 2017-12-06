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
from six.moves import range, zip
import threading

from .input_source_base import InputSource
from ..dataflow import DataFlow, MapData, RepeatedData, DataFlowTerminated
from ..tfutils.summary import add_moving_summary
from ..tfutils.common import get_op_tensor_name
from ..tfutils.tower import get_current_tower_context
from ..utils import logger
from ..utils.concurrency import ShareSessionThread
from ..utils.develop import log_deprecated
from ..callbacks.base import Callback, CallbackFactory
from ..callbacks.graph import RunOp

__all__ = ['PlaceholderInput', 'FeedInput',
           'FeedfreeInput',
           'QueueInput', 'BatchQueueInput',
           'DummyConstantInput', 'TensorInput',
           'TFDatasetInput',
           'StagingInputWrapper',
           'StagingInput']


def _get_reset_callback(df):
    return CallbackFactory(setup_graph=lambda _: df.reset_state())


class PlaceholderInput(InputSource):
    """
    Just produce placeholders as input tensors.
    """
    def __init__(self, prefix=''):
        """
        Args:
            prefix(str): an optional prefix to add to the placeholder.
        """
        self._prefix = prefix

    def _setup(self, inputs):
        self._all_placehdrs = [v.build_placeholder(prefix=self._prefix) for v in inputs]

    def _get_input_tensors(self):
        return self._all_placehdrs


class FeedInput(InputSource):
    """ Input by iterating over a DataFlow and feed datapoints. """

    class _FeedCallback(Callback):
        def __init__(self, ds, placeholders):
            self._ds = ds
            self._itr = self._ds.get_data()
            self._placeholders = placeholders

        def _before_run(self, _):
            dp = next(self._itr)
            assert len(dp) == len(self._placeholders), "[FeedInput] datapoints and inputs are of different length!"
            feed = dict(zip(self._placeholders, dp))
            return tf.train.SessionRunArgs(fetches=[], feed_dict=feed)

        def _reset(self):
            self._itr = self._ds.get_data()

    def __init__(self, ds, infinite=True):
        """
        Args:
            ds (DataFlow): the input DataFlow.
            infinite (bool): When set to False, will raise StopIteration when
                ds is exhausted.
        """
        assert isinstance(ds, DataFlow), ds
        self.ds = ds
        if infinite:
            self._iter_ds = RepeatedData(self.ds, -1)
        else:
            self._iter_ds = self.ds

    def _size(self):
        return self.ds.size()

    def _setup(self, inputs):
        # placeholders as input are always safe to reuse.
        self._all_placehdrs = [v.build_placeholder_reuse() for v in inputs]
        self._cb = self._FeedCallback(self._iter_ds, self._all_placehdrs)

    def _get_input_tensors(self):
        return self._all_placehdrs

    def _reset_state(self):
        self._cb._reset()

    def _get_callbacks(self):
        return [self._cb, _get_reset_callback(self._iter_ds)]


class FeedfreeInput(InputSource):
    """ Abstract base for input without feed,
    e.g. by queue or other operations. """

    def _reset_state(self):
        pass


# TODO enqueu_many? https://github.com/tensorflow/tensorflow/issues/7817#issuecomment-282053155
class EnqueueThread(ShareSessionThread):
    def __init__(self, queue, ds, placehdrs):
        super(EnqueueThread, self).__init__()
        self.name = 'EnqueueThread ' + queue.name
        self.daemon = True
        self.dataflow = ds
        self.queue = queue
        self.placehdrs = placehdrs

        self.op = self.queue.enqueue(self.placehdrs)
        self.close_op = self.queue.close(cancel_pending_enqueues=True)

        self._lock = threading.Lock()

    def run(self):
        with self.default_sess():
            try:
                self.reinitialize_dataflow()
                while True:
                    # pausable loop
                    self._lock.acquire()
                    self._lock.release()

                    dp = next(self._itr)
                    feed = dict(zip(self.placehdrs, dp))
                    self.op.run(feed_dict=feed)
            except (tf.errors.CancelledError, tf.errors.OutOfRangeError, DataFlowTerminated):
                pass
            except Exception as e:
                if isinstance(e, RuntimeError) and 'closed Session' in str(e):
                    pass
                else:
                    logger.exception("Exception in {}:".format(self.name))
            finally:
                try:
                    self.close_op.run()
                except Exception:
                    pass
                logger.info("{} Exited.".format(self.name))

    def reinitialize_dataflow(self):
        self._itr = self.dataflow.get_data()

    def pause(self):
        self._lock.acquire()

    def resume(self):
        self._lock.release()


class QueueInput(FeedfreeInput):
    """ Enqueue datapoints from a DataFlow to a TF queue.
        And the model receives dequeued tensors.

        Calling :meth:`reset_state()` will clear the queue and reset the dataflow.
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
        self._inf_ds = RepeatedData(ds, -1)

    def _size(self):
        return self.ds.size()

    def _setup(self, inputs):
        self._input_placehdrs = [v.build_placeholder_reuse() for v in inputs]
        assert len(self._input_placehdrs) > 0, \
            "QueueInput has to be used with some inputs!"
        with self.cached_name_scope():
            if self.queue is None:
                self.queue = tf.FIFOQueue(
                    50, [x.dtype for x in self._input_placehdrs],
                    name='input_queue')
            logger.info("Setting up the queue '{}' for CPU prefetching ...".format(self.queue.name))
            self.thread = EnqueueThread(self.queue, self._inf_ds, self._input_placehdrs)

            self._dequeue_op = self.queue.dequeue(name='dequeue_for_reset')

    def _reset_state(self):
        self.thread.pause()     # pause enqueue

        opt = tf.RunOptions()
        opt.timeout_in_ms = 2000   # 2s
        sess = tf.get_default_session()
        # dequeue until empty
        try:
            while True:
                sess.run(self._dequeue_op, options=opt)
        except tf.errors.DeadlineExceededError:
            pass

        # reset dataflow, start thread
        self.thread.reinitialize_dataflow()
        self.thread.resume()

    def _create_ema_callback(self):
        """
        Create a hook-only callback which maintain EMA of the queue size.
        Also tf.summary.scalar the EMA.
        """
        with self.cached_name_scope():
            # in TF there is no API to get queue capacity, so we can only summary the size
            size = tf.cast(self.queue.size(), tf.float32, name='queue_size')
        size_ema_op = add_moving_summary(size, collection=None)[0].op
        return RunOp(
            lambda: size_ema_op,
            run_before=False,
            run_as_trigger=False,
            run_step=True)

    def _get_callbacks(self):
        from ..callbacks.concurrency import StartProcOrThread
        cb = StartProcOrThread(self.thread)
        cb.chief_only = False
        return [cb, self._create_ema_callback(), _get_reset_callback(self._inf_ds)]

    def _get_input_tensors(self):
        with tf.device('/cpu:0'), self.cached_name_scope():
            ret = self.queue.dequeue(name='input_deque')
            if isinstance(ret, tf.Tensor):  # only one input
                ret = [ret]
            assert len(ret) == len(self._input_placehdrs)
            for qv, v in zip(ret, self._input_placehdrs):
                qv.set_shape(v.get_shape())
            return ret


class BatchQueueInput(QueueInput):
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
        super(BatchQueueInput, self).__init__(ds, queue)
        self.batch_size = int(batch_size)

    def _size(self):
        return self.ds.size() // self.batch_size

    def _setup(self, inputs):
        logger.info("Setting up the queue for CPU prefetching ...")
        self.input_placehdrs = [v.build_placeholder_reuse() for v in inputs]
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

        with self.cached_name_scope():
            if self.queue is None:
                self.queue = tf.FIFOQueue(
                    3000, [x.dtype for x in self.input_placehdrs],
                    shapes=shapes,
                    name='input_queue')
            for shp in self.queue.shapes:
                assert shp.is_fully_defined(), shape_err

            self.thread = EnqueueThread(self.queue, self._inf_ds, placehdrs_nobatch)

    def _get_input_tensors(self):
        with tf.device('/cpu:0'), self.cached_name_scope():
            ret = self.queue.dequeue_many(self.batch_size, name='input_deque')
            if isinstance(ret, tf.Tensor):  # only one input
                ret = [ret]
            assert len(ret) == len(self.input_placehdrs)
            for qv, v in zip(ret, self.input_placehdrs):
                shp = v.get_shape().as_list()
                shp[0] = self.batch_size
                qv.set_shape(shp)
            return ret


# TODO tensor inputs can be drained? look at the new dataset API.
class TensorInput(FeedfreeInput):
    """ Input from a list of tensors, e.g. a TF data reading pipeline.
        The PTB training example shows how to use it.
    """

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
        self._fixed_size = size

    def _setup(self, inputs_desc):
        self._desc = inputs_desc

    def _size(self):
        if self._fixed_size is None:
            raise NotImplementedError("size of TensorInput is undefined!")
        return self._fixed_size

    def _get_input_tensors(self):
        with self.cached_name_scope():
            ret = self.get_tensor_fn()
        assert len(ret) == len(self._desc), "{} != {}".format(len(ret), len(self._desc))
        return ret


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
            assert ctx is not None
            assert len(self.shapes) == len(self._desc)
            for idx, p in enumerate(self._desc):
                tlist.append(tf.constant(
                    0, dtype=p.type,
                    name='dummy-{}-{}'.format(p.name, ctx.index),
                    shape=self.shapes[idx]))
            return tlist
        super(DummyConstantInput, self).__init__(fn)


class ZMQInput(TensorInput):
    """
    Not well implemented yet. Don't use.
    """
    def __init__(self, endpoint):
        self._endpoint = endpoint

        from tensorpack.user_ops import zmq_recv

        def fn():
            ret = zmq_recv(self._endpoint, [x.dtype for x in self.inputs_desc])
            if isinstance(ret, tf.Tensor):
                ret = [ret]
            assert len(ret) == len(self.inputs_desc)
            for qv, v in zip(ret, self.inputs_desc):
                qv.set_shape(v.shape)
            return ret
        super(ZMQInput, self).__init__(fn)

    def _setup(self, inputs_desc):
        self.inputs_desc = inputs_desc
        assert len(self.inputs_desc) > 0, \
            "ZMQInput has to be used with InputDesc!"


class TFDatasetInput(FeedfreeInput):
    """
    Use a :class:`tf.contrib.data.Dataset` instance as input.

    Note:
        In training, the dataset should be infinite (use :func:`repeat()`).
    """
    def __init__(self, dataset):
        """
        Args:
            dataset (tf.contrib.data.Dataset):
        """
        self._dataset = dataset

    def _setup(self, inputs_desc):
        self._desc = inputs_desc
        types = self._dataset.output_types
        desc_types = tuple([k.type for k in inputs_desc])
        assert len(types) == len(desc_types), \
            "Dataset and InputDesc has different length! {} != {}".format(
                len(types), len(desc_types))
        assert types == desc_types, \
            "Types of dataset and InputDesc don't match! {} != {}".format(
                str(types), str(desc_types))
        shapes = self._dataset.output_shapes
        desc_shapes = [k.shape for k in inputs_desc]
        for idx, (s1, s2) in enumerate(zip(shapes, desc_shapes)):
            s2 = tf.TensorShape(s2)
            assert s2.is_compatible_with(s1), \
                "InputDesc '{}' has incompatible shape with dataset! {} vs {}".format(
                    inputs_desc[idx].name, s2, s1)
        self._iterator = self._dataset.make_initializable_iterator()
        self._init_op = self._iterator.initializer

    def _reset_state(self):
        self._init_op.run()

    def _get_input_tensors(self):
        desc_shapes = [k.shape for k in self._desc]
        ret = self._iterator.get_next()
        assert len(ret) == len(desc_shapes)
        for t, shp in zip(ret, desc_shapes):
            t.set_shape(shp)
        return ret

    @staticmethod
    def dataflow_to_dataset(df, types):
        """
        Wrap a dataflow to tf.data.Dataset.
        Will reset df.

        Args:
            df (DataFlow)
            types([tf.DType])
        """
        assert isinstance(df, DataFlow), df
        assert isinstance(types, (list, tuple)), types
        df = MapData(df, lambda dp: tuple(dp))
        df.reset_state()
        ds = tf.data.Dataset.from_generator(
            df.get_data, tuple(types))
        return ds


class StagingInput(FeedfreeInput):
    """
    A wrapper around a feedfree input,
    to prefetch the input in StagingArea (on GPUs).
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
            self._initialized = False

        def _prefill(self):
            logger.info("Pre-filling staging area ...")
            for k in range(self.nr_stage):
                self.stage_op.run()

        def _before_run(self, ctx):
            # This has to happen once, right before the first iteration.
            if not self._initialized:
                self._initialized = True
                self._prefill()
            return self.fetches

    def __init__(self, input, towers, nr_stage=5):
        """
        Args:
            input (FeedfreeInput):
            towers ([int]): list of GPU ids to prefetch on.
            nr_stage: number of elements to prefetch on each GPU.
        """
        assert isinstance(input, FeedfreeInput), input
        self._input = input
        if not isinstance(towers[0], int):
            # API changed
            log_deprecated("StagingInput(devices=)", "Use (towers=) instead!", "2018-01-31")
            self._devices = towers
        else:
            self._devices = ['/gpu:{}'.format(k) for k in towers]

        self._nr_stage = nr_stage
        self._areas = []
        self._stage_ops = []
        self._unstage_ops = []

    def _setup(self, inputs):
        self._input.setup(inputs)
        self._setup_staging_areas()

    def _get_callbacks(self):
        cbs = self._input.get_callbacks()

        cbs.append(
            StagingInput.StagingCallback(
                self._get_stage_op(), self._get_unstage_op(), self._nr_stage))
        return cbs

    def _setup_staging_areas(self):
        logger.info("Setting up StagingArea for GPU prefetching ...")
        with self.cached_name_scope():
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

    def _size(self):
        return self._input.size()

    def _get_input_tensors(self):
        ctx = get_current_tower_context()
        ret = self._unstage_ops[ctx.index]
        return ret

    def _get_stage_op(self):
        with self.cached_name_scope():
            return tf.group(*self._stage_ops)

    def _get_unstage_op(self):
        with self.cached_name_scope():
            all_outputs = list(chain.from_iterable(self._unstage_ops))
            return tf.group(*all_outputs)


StagingInputWrapper = StagingInput
