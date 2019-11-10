# -*- coding: utf-8 -*-
# File: inference_runner.py


import itertools
import sys
from contextlib import contextmanager
import tqdm
from tensorflow.python.training.monitored_session import _HookedSession as HookedSession

from ..compat import tfv1 as tf
from ..dataflow.base import DataFlow
from ..input_source import FeedInput, InputSource, QueueInput, StagingInput
from ..tfutils.tower import PredictTowerContext
from ..utils import logger
from ..utils.utils import get_tqdm_kwargs
from .base import Callback
from .group import Callbacks
from .inference import Inferencer

__all__ = ['InferenceRunnerBase', 'InferenceRunner',
           'DataParallelInferenceRunner']


def _device_from_int(dev):
    return '/gpu:{}'.format(dev) if dev >= 0 else '/cpu:0'


class InferencerToHook(tf.train.SessionRunHook):
    def __init__(self, inf, fetches):
        self._inf = inf
        self._fetches = fetches

    def before_run(self, _):
        return tf.train.SessionRunArgs(fetches=self._fetches)

    def after_run(self, _, run_values):
        self._inf.on_fetches(run_values.results)


@contextmanager
def _inference_context():
    msg = "You might need to check your input implementation."
    try:
        yield
    except (StopIteration, tf.errors.CancelledError):
        logger.error(
            "[InferenceRunner] input stopped before reaching its __len__()! " + msg)
        raise
    except tf.errors.OutOfRangeError:   # tf.data reaches an end
        pass


class InferenceRunnerBase(Callback):
    """ Base class for inference runner.

    Note:
        1. InferenceRunner will use `input.size()` to determine
           how much iterations to run, so you're responsible to ensure that
           `input.size()` is accurate.
        2. Only works with instances of `TowerTrainer`.
    """
    def __init__(self, input, infs):
        """
        Args:
            input (InputSource): the input to use. Must have an accurate ``size()``.
            infs (list[Inferencer]): list of :class:`Inferencer` to run.
        """
        self._input_source = input
        if not isinstance(infs, list):
            self.infs = [infs]
        else:
            self.infs = infs
        for v in self.infs:
            assert isinstance(v, Inferencer), v

        try:
            self._size = input.size()
        except NotImplementedError:
            self._size = 0

        self._hooks = []

    def register_hook(self, hook):
        """
        Args:
            hook (tf.train.SessionRunHook):
        """
        self._hooks.append(hook)

    def _before_train(self):
        self._hooked_sess = HookedSession(self.trainer.sess, self._hooks)
        self._input_callbacks.before_train()
        if self._size > 0:
            logger.info("[InferenceRunner] Will eval {} iterations".format(self._size))
        else:
            logger.warn("[InferenceRunner] Got an InputSource with unknown size! Will iterate until OutOfRangeError!")

    def _after_train(self):
        self._input_callbacks.after_train()


class InferenceRunner(InferenceRunnerBase):
    """
    A callback that runs a list of :class:`Inferencer` on some :class:`InputSource`.
    """

    def __init__(self, input, infs, tower_name='InferenceTower', tower_func=None, device=0):
        """
        Args:
            input (InputSource or DataFlow): The :class:`InputSource` to run
                inference on.  If given a DataFlow, will use :class:`FeedInput`.
            infs (list): a list of :class:`Inferencer` instances.
            tower_name (str): the name scope of the tower to build.
                If multiple InferenceRunner are used, each needs a different tower_name.
            tower_func (tfutils.TowerFunc or None): the tower function to be used to build the graph.
                By defaults to call `trainer.tower_func` under a `training=False` TowerContext,
                but you can change it to a different tower function
                if you need to inference with several different graphs.
            device (int): the device to use
        """
        if isinstance(input, DataFlow):
            # use infinite=False so that a dataflow without size will stop normally
            # TODO a better way to handle inference size
            input = FeedInput(input, infinite=False)
        assert isinstance(input, InputSource), input
        assert not isinstance(input, StagingInput), input
        self._tower_name = tower_name
        self._device_id = device
        self._device = _device_from_int(device)
        self._tower_func = tower_func
        super(InferenceRunner, self).__init__(input, infs)

    def _build_hook(self, inf):
        out_names = inf.get_fetches()
        fetches = self._tower_handle.get_tensors(out_names)
        return InferencerToHook(inf, fetches)

    def _setup_graph(self):
        if self._tower_func is None:
            assert self.trainer.tower_func is not None, "You must set tower_func of the trainer to use InferenceRunner!"
            self._tower_func = self.trainer.tower_func
        input_callbacks = self._input_source.setup(self._tower_func.input_signature)

        vs_name = self.trainer._vs_name_for_predictor(self._device_id)
        logger.info("[InferenceRunner] Building tower '{}' on device {} {}...".format(
            self._tower_name, self._device,
            "with variable scope '{}'".format(vs_name) if vs_name else ''))
        with tf.variable_scope(tf.get_variable_scope(), reuse=True), \
                tf.device(self._device), \
                PredictTowerContext(self._tower_name, vs_name=vs_name):
            self._tower_func(*self._input_source.get_input_tensors())
            self._tower_handle = self._tower_func.towers[-1]

        for h in [self._build_hook(inf) for inf in self.infs]:
            self.register_hook(h)
        # trigger_{step,epoch}, {before,after}_epoch is ignored.
        # We assume that InputSource callbacks won't use these methods
        self._input_callbacks = Callbacks(input_callbacks)
        for h in self._input_callbacks.get_hooks():
            self.register_hook(h)

        for inf in self.infs:
            inf.setup_graph(self.trainer)
        self._input_callbacks.setup_graph(self.trainer)

    def _trigger(self):
        for inf in self.infs:
            inf.before_epoch()

        self._input_source.reset_state()
        # iterate over the data, and run the hooked session
        with _inference_context(), \
                tqdm.tqdm(total=self._size, **get_tqdm_kwargs()) as pbar:
            num_itr = self._size if self._size > 0 else sys.maxsize
            for _ in range(num_itr):
                self._hooked_sess.run(fetches=[])
                pbar.update()
        for inf in self.infs:
            inf.trigger_epoch()


class DataParallelInferenceRunner(InferenceRunnerBase):
    """
    Inference with data-parallel support on multiple GPUs.
    It will build one predict tower on each GPU, and run prediction
    with a large total batch in parallel on all GPUs.
    It will run the remainder (when the total size of input is not a multiple of #GPU)
    sequentially.
    """
    def __init__(self, input, infs, gpus, tower_name='InferenceTower', tower_func=None):
        """
        Args:
            input (DataFlow or QueueInput)
            gpus (int or list[int]): #gpus, or list of GPU id
            tower_name (str): the name scope of the tower to build.
                If multiple InferenceRunner are used, each needs a different tower_name.
            tower_func (tfutils.TowerFunc or None): the tower function to be used to build the graph.
                The tower function will be called under a `training=False` TowerContext.
                The default is `trainer.tower_func`,
                but you can change it to a different tower function
                if you need to inference with several different models.
        """
        if isinstance(gpus, int):
            gpus = list(range(gpus))
        self._devices = [_device_from_int(k) for k in gpus]
        self._tower_names = ['{}{}'.format(tower_name, k) for k in range(len(gpus))]

        if isinstance(input, DataFlow):
            input = QueueInput(input)
        assert isinstance(input, QueueInput), input
        super(DataParallelInferenceRunner, self).__init__(input, infs)
        assert self._size > 0, "Input for DataParallelInferenceRunner must have a size!"

        self._hooks = []
        self._hooks_parallel = []
        self._tower_func = tower_func

    def _setup_graph(self):
        self._handles = []
        if self._tower_func is None:
            assert self.trainer.tower_func is not None, "You must set tower_func of the trainer to use InferenceRunner!"
            self._tower_func = self.trainer.tower_func

        input_callbacks = self._input_source.setup(self._tower_func.input_signature)
        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
            for idx, dev in enumerate(self._devices):
                vs_name = self.trainer._vs_name_for_predictor(idx)
                with tf.device(dev), PredictTowerContext(
                        self._tower_names[idx], vs_name=vs_name):
                    logger.info("[InferenceRunner] Building tower '{}' on device {} {}...".format(
                        self._tower_names[idx], dev,
                        "with variable scope '{}'".format(vs_name) if vs_name else ''))
                    # TODO log for tower creation, here or in tower.py?
                    self._tower_func(*self._input_source.get_input_tensors())
                    self._handles.append(self._tower_func.towers[-1])

        # setup callbacks and hooks
        self._input_callbacks = Callbacks(input_callbacks)

        # TODO InputSource might have hooks which break us.
        # e.g. hooks from StagingInput will force the consumption
        # of nr_tower datapoints in every run.
        input_hooks = self._input_callbacks.get_hooks()
        self._hooks.extend([self._build_hook(inf) for inf in self.infs] + input_hooks)
        self._hooks_parallel.extend([self._build_hook_parallel(inf) for inf in self.infs] + input_hooks)

        for inf in self.infs:
            inf.setup_graph(self.trainer)
        self._input_callbacks.setup_graph(self.trainer)

    def register_hook(self, h):
        logger.info(
            "[DataParallelInferenceRunner] Registering hook {} on both parallel and sequential inference.")
        self._hooks.append(h)
        self._hooks_parallel.append(h)

    class _InferencerToHookDataParallel(InferencerToHook):
        def __init__(self, inf, fetches, size):
            """
            Args:
                size(int): number of tensors to fetch per tower
            """
            super(DataParallelInferenceRunner._InferencerToHookDataParallel, self).__init__(inf, fetches)
            assert len(self._fetches) % size == 0
            self._sz = size

        def after_run(self, _, run_values):
            res = run_values.results
            for i in range(0, len(res), self._sz):
                vals = res[i:i + self._sz]
                self._inf.on_fetches(vals)

    def _build_hook_parallel(self, inf):
        out_names = inf.get_fetches()
        sz = len(out_names)
        fetches = list(itertools.chain(*[t.get_tensors(out_names) for t in self._handles]))
        return self._InferencerToHookDataParallel(inf, fetches, sz)

    def _build_hook(self, inf):
        out_names = inf.get_fetches()
        fetches = self._handles[0].get_tensors(out_names)
        return InferencerToHook(inf, fetches)

    def _before_train(self):
        super(DataParallelInferenceRunner, self)._before_train()
        self._parallel_hooked_sess = HookedSession(self.trainer.sess, self._hooks_parallel)

    def _trigger(self):
        for inf in self.infs:
            inf.before_epoch()

        total = self._size
        nr_tower = len(self._devices)
        self._input_source.reset_state()
        with _inference_context():
            with tqdm.tqdm(total=total, **get_tqdm_kwargs()) as pbar:
                while total >= nr_tower:
                    self._parallel_hooked_sess.run(fetches=[])
                    pbar.update(nr_tower)
                    total -= nr_tower
                # take care of the rest
                for _ in range(total):
                    self._hooked_sess.run(fetches=[])
                    pbar.update(1)
        for inf in self.infs:
            inf.trigger_epoch()
