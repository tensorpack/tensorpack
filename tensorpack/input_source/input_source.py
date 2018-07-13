# -*- coding: utf-8 -*-
# File: input_source.py


import tensorflow as tf
try:
    from tensorflow.python.ops.data_flow_ops import StagingArea
except ImportError:
    pass

from contextlib import contextmanager
from itertools import chain
from six.moves import range, zip
import threading

from .input_source_base import InputSource
from ..dataflow import DataFlow, MapData, RepeatedData, DataFlowTerminated
from ..tfutils.summary import add_moving_summary
from ..tfutils.common import get_op_tensor_name
from ..tfutils.tower import get_current_tower_context
from ..tfutils.dependency import dependency_of_fetches
from ..utils import logger
from ..utils.concurrency import ShareSessionThread
from ..utils.develop import log_deprecated, deprecated
from ..callbacks.base import Callback, CallbackFactory
from ..callbacks.graph import RunOp

__all__ = ['PlaceholderInput', 'FeedInput', 'FeedfreeInput',
           'QueueInput', 'BatchQueueInput',
           'DummyConstantInput', 'TensorInput',
           'ZMQInput', 'TFDatasetInput',
           'StagingInputWrapper', 'StagingInput']


def _get_reset_callback(df):
    return CallbackFactory(setup_graph=lambda _: df.reset_state())


class PlaceholderInput(InputSource):
    """
    Just produce placeholders as input tensors.
    """
    def _setup(self, inputs):
        self._all_placehdrs = [v.build_placeholder_reuse() for v in inputs]

    def _get_input_tensors(self):
        return self._all_placehdrs


class FeedInput(InputSource):
    """
    Input by iterating over a DataFlow and feed datapoints.

    Note:
        If `get_input_tensors()` is called more than one time, it will return the same placeholders (i.e. feed points)
        as the first time.
        Therefore you can't use it for data-parallel training.
    """

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


# TODO enqueue_many? https://github.com/tensorflow/tensorflow/issues/7817#issuecomment-282053155
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

        self._running = threading.Event()
        self._running.set()
        # self._size = queue.size()

    def run(self):
        with self.default_sess():
            try:
                self.reinitialize_dataflow()
                while True:
                    # pausable loop
                    if not self._running.is_set():
                        self._running.wait()

                    dp = next(self._itr)
                    feed = dict(zip(self.placehdrs, dp))
                    # _, sz = sess.run([self.op, self._sz], feed_dict=feed)
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
        self._running.clear()

    def resume(self):
        self._running.set()


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
        self._started = False

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

    def refill_queue(self):
        """
        Clear the queue, then call dataflow.get_data() again and fill into the queue.
        """
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
        size_ema_op = add_moving_summary(size, collection=None, decay=0.5)[0].op
        return RunOp(
            lambda: size_ema_op,
            run_before=False,
            run_as_trigger=False,
            run_step=True)

    def _get_callbacks(self):
        from ..callbacks.concurrency import StartProcOrThread
        cb = StartProcOrThread(self.thread)
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
    """ Input with a constant zero tensor placed on GPU.
        Useful for debugging performance issues """
    def __init__(self, shapes):
        """
        Args:
            shapes (list[list]): a list of fully-specified shapes.
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
    Receive tensors from a ZMQ endpoint, with ops from https://github.com/tensorpack/zmq_ops.
    It works with :meth:`dataflow.remote.send_dataflow_zmq(format='zmq_ops')`.
    """
    def __init__(self, end_point, hwm, bind=True):
        """
        Args:
            end_point (str):
            hwm (int):
        """
        self._end_point = end_point
        self._hwm = int(hwm)
        self._bind = bind

        def fn():
            ret = self._zmq_pull_socket.pull()
            assert len(ret) == len(self._desc)
            for qv, v in zip(ret, self._desc):
                qv.set_shape(v.shape)
            return ret
        super(ZMQInput, self).__init__(fn)

    def _setup(self, inputs_desc):
        assert len(inputs_desc) > 0, \
            "ZMQInput has to be used with InputDesc!"
        self._desc = inputs_desc

        import zmq_ops
        self._zmq_pull_socket = zmq_ops.ZMQPullSocket(
            self._end_point,
            [x.type for x in inputs_desc],
            hwm=self._hwm,
            bind=self._bind)


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
        Will also reset the dataflow.

        If for training, you'll need to add `.repeat()` on the returned
        dataset, if the dataflow iterator can terminate.

        Args:
            df (DataFlow)
            types([tf.DType])

        Returns:
            (tf.data.Dataset)
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

    It works by registering hooks to put & get tensors into the StagingArea.
    If `get_input_tensors` gets called multiple times,
    it requires that all outputs ever produced by this InputSource will be fetched together.

    This means that in multi-GPU training, you should ensure that each call on `hooked_sess.run`
    depends on either all input tensors on all GPUs, or no input tensors at all.
    As a result you cannot use this InputSource for :class:`InferenceRunner`.
    """
    class StagingCallback(Callback):
        """
        A callback registered by this input source, to make sure stage/unstage
        is run at each step.
        """
        def __init__(self, input, nr_stage):
            self.nr_stage = nr_stage
            self._input = input
            self._initialized = False

        def _setup_graph(self):
            self.stage_op = self._input._get_stage_op()
            unstage_ops = self._input._get_unstage_ops()
            unstage_op = tf.group(*unstage_ops, name='unstage_all')
            self._check_dependency_op = unstage_ops[0]
            self.fetches = tf.train.SessionRunArgs(
                fetches=[self.stage_op, unstage_op])

        def _prefill(self):
            logger.info("Pre-filling StagingArea ...")
            for k in range(self.nr_stage):
                self.stage_op.run()
            logger.info("{} element{} put into StagingArea.".format(
                self.nr_stage, "s were" if self.nr_stage > 1 else " was"))

        def _before_run(self, ctx):
            # This has to happen once, right before the first iteration.
            # doing it in `before_train` may not work because QueueInput happens in before_train.
            if not self._initialized:
                self._initialized = True
                self._prefill()
            # Only step the stagingarea when the input is evaluated in this sess.run
            fetches = ctx.original_args.fetches
            if dependency_of_fetches(fetches, self._check_dependency_op):
                return self.fetches

    def __init__(self, input, towers=None, nr_stage=1, device=None):
        """
        Args:
            input (FeedfreeInput):
            nr_stage: number of elements to prefetch into each StagingArea, at the beginning.
                Since enqueue and dequeue are synchronized, prefetching 1 element should be sufficient.
            towers: deprecated
            device (str or None): if not None, place the StagingArea on a specific device. e.g., '/cpu:0'.
                Otherwise, they are placed under where `get_inputs_tensors`
                gets called, which could be unspecified in case of simple trainers.
        """
        assert isinstance(input, FeedfreeInput), input
        self._input = input
        if towers is not None:
            log_deprecated("StagingInput(towers=)", "Devices are handled automatically.", "2018-03-31")

        self._nr_stage = nr_stage
        self._areas = []
        self._stage_ops = []
        self._unstage_ops = []
        self._device = device

    def _setup(self, inputs):
        self._input.setup(inputs)
        with self.cached_name_scope():
            pass    # just to cache the correct ns to use

    def _get_callbacks(self):
        cbs = self._input.get_callbacks()

        # this callback has to happen after others, so StagingInput can be stacked together
        cbs.append(
            StagingInput.StagingCallback(self, self._nr_stage))
        return cbs

    def _size(self):
        return self._input.size()

    @contextmanager
    def _device_ctx(self):
        if not self._device:
            yield
        else:
            with tf.device(self._device):
                yield

    def _get_input_tensors(self):
        inputs = self._input.get_input_tensors()

        with self._device_ctx():
            with self.cached_name_scope():
                # Putting variables to stagingarea will cause trouble
                dtypes = []
                for idx in range(len(inputs)):
                    dtype = inputs[idx].dtype
                    if dtype.base_dtype != dtype:     # is reference type
                        inputs[idx] = tf.identity(inputs[idx])
                    dtypes.append(dtype.base_dtype)

                # TODO tensorflow/benchmarks use static shapes here,
                # though it doesn't seem to help. We can use it when it's known.
                stage = StagingArea(dtypes, shapes=None)

            # put & get automatically inherit the name scope from the area
            self._stage_ops.append(stage.put(inputs))
            self._areas.append(stage)
            outputs = stage.get()
            if isinstance(outputs, tf.Tensor):  # when size=1, TF doesn't return a list
                outputs = [outputs]

            for vin, vout in zip(inputs, outputs):
                vout.set_shape(vin.get_shape())
            self._unstage_ops.append(outputs)
            # self._size_ops.append(stage.size())
            return outputs

    def _get_stage_op(self):
        with self.cached_name_scope():
            return tf.group(*self._stage_ops)

    def _get_unstage_ops(self):
        with self.cached_name_scope():
            all_outputs = list(chain.from_iterable(self._unstage_ops))
            return all_outputs

    # for debugging only
    def _create_ema_callback(self):
        def create_ema_op():
            with self.cached_name_scope():
                avg_size = tf.truediv(tf.add_n(self._size_ops), len(self._size_ops), name='avg_stagingarea_size')
                return add_moving_summary(avg_size, collection=None)[0].op
        return RunOp(
            create_ema_op,
            run_before=False,
            run_as_trigger=False,
            run_step=True)


@deprecated("Renamed to StagingInput", "2018-08-01")
def StagingInputWrapper(*args, **kwargs):
    return StagingInput(*args, **kwargs)
