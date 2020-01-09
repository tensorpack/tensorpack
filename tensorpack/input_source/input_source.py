# -*- coding: utf-8 -*-
# File: input_source.py


import threading
from contextlib import contextmanager
from itertools import chain
import tensorflow as tf

from ..compat import tfv1
from ..callbacks.base import Callback, CallbackFactory
from ..callbacks.graph import RunOp
from ..dataflow import DataFlow, MapData, RepeatedData, DataFlowTerminated
from ..tfutils.common import get_op_tensor_name
from ..tfutils.dependency import dependency_of_fetches
from ..tfutils.summary import add_moving_summary
from ..tfutils.tower import get_current_tower_context
from ..utils import logger
from ..utils.concurrency import ShareSessionThread
from .input_source_base import InputSource, build_or_reuse_placeholder

try:
    from tensorflow.python.ops.data_flow_ops import StagingArea
except ImportError:
    pass


__all__ = ['PlaceholderInput', 'FeedInput', 'FeedfreeInput',
           'QueueInput', 'BatchQueueInput',
           'DummyConstantInput', 'TensorInput',
           'ZMQInput', 'TFDatasetInput',
           'StagingInput']


def _get_reset_callback(df):
    return CallbackFactory(setup_graph=lambda _: df.reset_state())


def _make_feeds(placeholders, datapoint):
    assert len(datapoint) == len(placeholders), \
        "Size of datapoint and placeholders are different: {} != {}".format(
            len(datapoint), len(placeholders))

    if isinstance(datapoint, (list, tuple)):
        return dict(zip(placeholders, datapoint))
    elif isinstance(datapoint, dict):
        ret = {p: datapoint[p.op.name] for p in placeholders}
        return ret
    else:
        raise TypeError("Got a datapoint of type {}!".format(type(datapoint)))


class PlaceholderInput(InputSource):
    """
    Just produce placeholders as input tensors.
    """
    def __init__(self):
        pass

    def _setup(self, inputs):
        self._all_placehdrs = [build_or_reuse_placeholder(v) for v in inputs]

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
            self._itr = self._ds.__iter__()
            self._placeholders = placeholders

        def _before_run(self, _):
            dp = next(self._itr)
            assert len(dp) == len(self._placeholders), "[FeedInput] datapoints and inputs are of different length!"
            feed = _make_feeds(self._placeholders, dp)
            return tfv1.train.SessionRunArgs(fetches=[], feed_dict=feed)

        def _reset(self):
            self._itr = self._ds.__iter__()

    def __init__(self, ds, infinite=True):
        """
        Args:
            ds (DataFlow): the input DataFlow.
            infinite (bool): When set to False, will raise StopIteration when
                ds is exhausted.
        """
        if not isinstance(ds, DataFlow):
            raise ValueError("FeedInput takes a DataFlow! Got {}".format(ds))
        self.ds = ds
        if infinite:
            self._iter_ds = RepeatedData(self.ds, -1)
        else:
            self._iter_ds = self.ds

    def _size(self):
        return len(self.ds)

    def _setup(self, inputs):
        # placeholders as input are always safe to reuse.
        self._all_placehdrs = [build_or_reuse_placeholder(v) for v in inputs]
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
                    feed = _make_feeds(self.placehdrs, dp)
                    # _, sz = sess.run([self.op, self._sz], feed_dict=feed)
                    self.op.run(feed_dict=feed)
            except (tf.errors.CancelledError, tf.errors.OutOfRangeError):
                pass
            except DataFlowTerminated:
                logger.info("[EnqueueThread] DataFlow has terminated.")
            except Exception as e:
                if isinstance(e, RuntimeError) and 'closed Session' in str(e):
                    pass
                else:
                    logger.exception("[EnqueueThread] Exception in thread {}:".format(self.name))
            finally:
                try:
                    self.close_op.run()
                except Exception:
                    pass
                logger.info("[EnqueueThread] Thread {} Exited.".format(self.name))

    def reinitialize_dataflow(self):
        self._itr = self.dataflow.__iter__()

    def pause(self):
        self._running.clear()

    def resume(self):
        self._running.set()


class QueueInput(FeedfreeInput):
    """ Enqueue datapoints from a DataFlow to a TF queue.
        And the model receives dequeued tensors.
    """

    def __init__(self, ds, queue=None):
        """
        Args:
            ds(DataFlow): the input DataFlow.
            queue (tf.QueueBase): A :class:`tf.QueueBase` whose type
                should match the corresponding input signature of the model.
                Defaults to a FIFO queue of size 50.
        """
        if not isinstance(ds, DataFlow):
            raise ValueError("QueueInput takes a DataFlow! Got {}".format(ds))
        self.queue = queue
        self.ds = ds
        self._inf_ds = RepeatedData(ds, -1)
        self._started = False

    def _size(self):
        return len(self.ds)

    def _setup(self, inputs):
        self._input_placehdrs = [build_or_reuse_placeholder(v) for v in inputs]
        assert len(self._input_placehdrs) > 0, \
            "QueueInput has to be used with some inputs!"
        with self.cached_name_scope():
            if self.queue is None:
                self.queue = tfv1.FIFOQueue(
                    50, [x.dtype for x in self._input_placehdrs],
                    name='input_queue')
            logger.info("Setting up the queue '{}' for CPU prefetching ...".format(self.queue.name))
            self.thread = EnqueueThread(self.queue, self._inf_ds, self._input_placehdrs)

            self._dequeue_op = self.queue.dequeue(name='dequeue_for_reset')

    def refill_queue(self):
        """
        Clear the queue, then call dataflow.__iter__() again and fill into the queue.
        """
        self.thread.pause()     # pause enqueue

        opt = tfv1.RunOptions()
        opt.timeout_in_ms = 2000   # 2s
        sess = tfv1.get_default_session()
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
        ret = RunOp(
            lambda: size_ema_op,
            run_before=False,
            run_as_trigger=False,
            run_step=True)
        ret.name_scope = "InputSource/EMA"
        return ret

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
                should match the corresponding input signature of the model.
                Defaults to a FIFO queue of size 3000.
        """
        super(BatchQueueInput, self).__init__(ds, queue)
        self.batch_size = int(batch_size)

    def _size(self):
        return len(self.ds) // self.batch_size

    def _setup(self, inputs):
        logger.info("Setting up the queue for CPU prefetching ...")
        self.input_placehdrs = [build_or_reuse_placeholder(v) for v in inputs]
        assert len(self.input_placehdrs) > 0, \
            "BatchQueueInput has to be used with some input signature!"

        # prepare placeholders without the first dimension
        placehdrs_nobatch = []
        for p in self.input_placehdrs:
            placehdrs_nobatch.append(tfv1.placeholder(
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
    """ Use inputs from a list of tensors, e.g. a TF data reading pipeline.
        The PTB training example shows how to use it.
    """

    def __init__(self, get_tensor_fn, size=None):
        """
        Args:
            get_tensor_fn ( -> [tf.Tensor]): a function which returns a list of input tensors
                (for example, [image, label]) when called.
                It will be called under a TowerContext and should return the inputs to be used in that tower.
                The returned tensors will be evaluated every iteration, it's your job to make sure it's possible.
            size(int): size of this input. Use None to leave it undefined.
        """
        if not callable(get_tensor_fn):
            raise ValueError("get_tensor_fn has to be a function! Got {}".format(get_tensor_fn))
        self.get_tensor_fn = get_tensor_fn
        if size is not None:
            size = int(size)
            assert size > 0
        self._fixed_size = size

    def _setup(self, input_signature):
        self._spec = input_signature

    def _size(self):
        if self._fixed_size is None:
            raise NotImplementedError("size of TensorInput is undefined!")
        return self._fixed_size

    def _get_input_tensors(self):
        with self.cached_name_scope():
            ret = self.get_tensor_fn()
        assert isinstance(ret, (list, tuple)), "get_tensor_fn needs to return a list!"
        assert len(ret) == len(self._spec), \
            "get_tensor_fn returns {} tensors but there are {} inputs".format(len(ret), len(self._spec))
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
            assert len(self.shapes) == len(self._spec)
            for idx, p in enumerate(self._spec):
                tlist.append(tf.constant(
                    0, dtype=p.dtype,
                    name='dummy-{}-{}'.format(p.name, ctx.index),
                    shape=self.shapes[idx]))
            return tlist
        super(DummyConstantInput, self).__init__(fn)


class ZMQInput(TensorInput):
    """
    Receive tensors from a ZMQ endpoint, with ops from https://github.com/tensorpack/zmq_ops.
    It works with :func:`dataflow.remote.send_dataflow_zmq(format='zmq_ops')`.
    """
    def __init__(self, end_point, hwm, bind=True):
        """
        Args:
            end_point (str): the ZMQ endpoint
            hwm (int): the ZMQ high-water-mark
        """
        self._end_point = end_point
        self._hwm = int(hwm)
        self._bind = bind

        def fn():
            ret = self._zmq_pull_socket.pull()
            assert len(ret) == len(self._spec)
            for qv, v in zip(ret, self._spec):
                qv.set_shape(v.shape)
            return ret
        super(ZMQInput, self).__init__(fn)

    def _setup(self, input_signature):
        super(ZMQInput, self)._setup(input_signature)
        assert len(input_signature) > 0, \
            "ZMQInput has to be used with input signature!"

        import zmq_ops
        self._zmq_pull_socket = zmq_ops.ZMQPullSocket(
            self._end_point,
            [x.dtype for x in input_signature],
            hwm=self._hwm,
            bind=self._bind)


class TFDatasetInput(FeedfreeInput):
    """
    Use a :class:`tf.data.Dataset` instance as input.

    Note:
        1. In training, the given dataset or dataflow has to be infinite
            (you can use :func:`repeat()`, or :class:`RepeatedData` ).

        2. TensorFlow may keep the dataflow alive even if the dataset is no
           longer used.
    """
    def __init__(self, dataset):
        """
        Args:
            dataset (tf.data.Dataset or DataFlow):
        """
        if isinstance(dataset, tf.data.Dataset):
            self._dataset = dataset
            self._dataflow = None
        elif isinstance(dataset, DataFlow):
            self._dataset = None
            self._dataflow = dataset
        else:
            raise ValueError("TFDatasetInput takes a tf.data.Dataset or DataFlow! Got {}".format(dataset))

    def _setup(self, input_signature):
        self._spec = input_signature
        if self._dataset is not None:
            types = self._dataset.output_types
            spec_types = tuple(k.dtype for k in input_signature)
            assert len(types) == len(spec_types), \
                "Dataset and input signature have different length! {} != {}".format(
                    len(types), len(spec_types))
            assert types == spec_types, \
                "Data types of dataset and input signature don't match! {} != {}".format(
                    str(types), str(spec_types))

            shapes = self._dataset.output_shapes
            spec_shapes = [k.shape for k in input_signature]
            for idx, (s1, s2) in enumerate(zip(shapes, spec_shapes)):
                s2 = tf.TensorShape(s2)
                assert s2.is_compatible_with(s1), \
                    "Input signature '{}' has incompatible shape with dataset! {} vs {}".format(
                        input_signature[idx].name, s2, s1)
        else:
            self._dataset = TFDatasetInput.dataflow_to_dataset(self._dataflow, [x.dtype for x in input_signature])

        self._iterator = self._dataset.make_initializable_iterator()
        self._init_op = self._iterator.initializer

    def _reset_state(self):
        self._init_op.run()

    def _get_input_tensors(self):
        spec_shapes = [k.shape for k in self._spec]
        ret = self._iterator.get_next()
        assert len(ret) == len(spec_shapes), \
            "Dataset returns {} tensors but there are {} inputs!".format(len(ret), len(spec_shapes))
        for t, shp in zip(ret, spec_shapes):
            t.set_shape(shp)
        return ret

    @staticmethod
    def dataflow_to_dataset(df, types):
        """
        Wrap a dataflow to tf.data.Dataset.
        This function will also reset the dataflow.

        If the dataflow itself is finite, the returned dataset is also finite.
        Therefore, if used for training, you'll need to add `.repeat()` on the returned
        dataset.

        Args:
            df (DataFlow): a dataflow which produces lists
            types([tf.DType]): list of types

        Returns:
            (tf.data.Dataset)

        Note:
            TensorFlow may keep the dataflow alive even if the dataset is no
            longer used.
        """
        # TODO theoretically it can support dict
        assert isinstance(df, DataFlow), df
        assert isinstance(types, (list, tuple)), types
        df = MapData(df, tuple)
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

    More than one StagingInput cannot be used together.
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
            self.fetches = tfv1.train.SessionRunArgs(
                fetches=[self.stage_op, unstage_op])

        def _prefill(self, sess):
            logger.info("Pre-filling StagingArea ...")
            for _ in range(self.nr_stage):
                self.stage_op.run(session=sess)
            logger.info("{} element{} put into StagingArea on each tower.".format(
                self.nr_stage, "s were" if self.nr_stage > 1 else " was"))

        def _before_run(self, ctx):
            # This has to happen once, right before the first iteration.
            # doing it in `before_train` may not work because QueueInput happens in before_train.
            if not self._initialized:
                self._initialized = True
                self._prefill(ctx.session)
            # Only step the stagingarea when the input is evaluated in this sess.run
            fetches = ctx.original_args.fetches
            if dependency_of_fetches(fetches, self._check_dependency_op):
                # note: this disable nesting of StagingInput
                return self.fetches

    def __init__(self, input, nr_stage=1, device=None):
        """
        Args:
            input (FeedfreeInput):
            nr_stage (int): number of elements to prefetch into each StagingArea, at the beginning.
                Since enqueue and dequeue are synchronized, prefetching 1 element should be sufficient.
            device (str or None): if not None, place the StagingArea on a specific device. e.g., '/cpu:0'.
                Otherwise, they are placed under where `get_inputs_tensors`
                gets called, which could be unspecified in case of simple trainers.
        """
        if not isinstance(input, FeedfreeInput):
            raise ValueError("StagingInput takes a FeedfreeInput! Got {}".format(input))
        if isinstance(input, StagingInput):
            raise ValueError("StagingInput cannot be nested!")

        self._input = input

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
                # Setting capacity to 1 to potentially save some memory, because we should
                # expect the consumers to run slower than the producer.
                stage = StagingArea(dtypes, shapes=None, capacity=1)

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
