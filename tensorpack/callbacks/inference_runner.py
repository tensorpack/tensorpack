#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: inference_runner.py


import sys
import tensorflow as tf
from tensorflow.python.training.monitored_session \
    import _HookedSession as HookedSession

import itertools
from contextlib import contextmanager
import tqdm
from six.moves import range

from ..utils import logger
from ..utils.utils import get_tqdm_kwargs
from ..dataflow.base import DataFlow

from ..input_source import (
    InputSource, FeedInput, QueueInput, StagingInput)
from ..graph_builder.predict import SimplePredictBuilder

from .base import Callback
from .group import Callbacks
from .inference import Inferencer

__all__ = ['InferenceRunnerBase', 'InferenceRunner',
           'DataParallelInferenceRunner']


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
            "[InferenceRunner] input stopped before reaching its size()! " + msg)
        raise
    except tf.errors.OutOfRangeError:   # tf.data reaches an end
        pass


class InferenceRunnerBase(Callback):
    """ Base class for inference runner.
        Please note that InferenceRunner will use `input.size()` to determine
        how much iterations to run, so you're responsible to ensure that
        `size()` is accurate.

        Also, InferenceRunner assumes that `trainer.model` exists.
    """
    def __init__(self, input, infs):
        """
        Args:
            input (InputSource): the input to use. Must have ``size()``.
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
            logger.info("InferenceRunner will eval {} iterations".format(self._size))
        else:
            logger.warn("InferenceRunner got an input with unknown size! It will iterate until OutOfRangeError!")

    def _after_train(self):
        self._input_callbacks.after_train()


class InferenceRunner(InferenceRunnerBase):
    """
    A callback that runs a list of :class:`Inferencer` on some :class:`InputSource`.
    """

    def __init__(self, input, infs, tower_name='InferenceTower', device=0):
        """
        Args:
            input (InputSource or DataFlow): The :class:`InputSource` to run
                inference on.  If given a DataFlow, will use :class:`FeedInput`.
            infs (list): a list of :class:`Inferencer` instances.
            tower_name (str): the name scope of the tower to build. Need to set a
                different one if multiple InferenceRunner are used.
            device (int): the device to use
        """
        if isinstance(input, DataFlow):
            input = FeedInput(input, infinite=True)     # TODO a better way to handle inference size
        assert isinstance(input, InputSource), input
        assert not isinstance(input, StagingInput), input
        self._tower_name = tower_name
        self._device = device
        super(InferenceRunner, self).__init__(input, infs)

    def _build_hook(self, inf):
        out_names = inf.get_fetches()
        fetches = self._tower_handle.get_tensors(out_names)
        return InferencerToHook(inf, fetches)

    def _setup_graph(self):
        device = self._device
        assert self.trainer.tower_func is not None, "You must set tower_func of the trainer to use InferenceRunner!"
        input_callbacks = self._input_source.setup(self.trainer.inputs_desc)

        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
            SimplePredictBuilder(
                ns_name=self._tower_name,
                vs_name=self.trainer._main_tower_vs_name, device=device).build(
                    self._input_source, self.trainer.tower_func)
            self._tower_handle = self.trainer.tower_func.towers[-1]

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
    def __init__(self, input, infs, gpus):
        """
        Args:
            input (DataFlow or QueueInput)
            gpus (int or list[int]): #gpus, or list of GPU id
        """
        if isinstance(gpus, int):
            gpus = list(range(gpus))
        self._tower_names = ['InferenceTower{}'.format(k) for k in range(len(gpus))]
        if isinstance(input, DataFlow):
            input = QueueInput(input)
        assert isinstance(input, QueueInput), input
        super(DataParallelInferenceRunner, self).__init__(input, infs)
        assert self._size > 0, "Input for DataParallelInferenceRunner must have a size!"
        self._gpus = gpus

        self._hooks = []
        self._hooks_parallel = []

    def _setup_graph(self):
        self._handles = []

        assert self.trainer.tower_func is not None, "You must set tower_func of the trainer to use InferenceRunner!"
        input_callbacks = self._input_source.setup(self.trainer.inputs_desc)
        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
            for idx, t in enumerate(self._gpus):
                tower_name = self._tower_names[idx]
                SimplePredictBuilder(
                    ns_name=tower_name,
                    vs_name=self.trainer._main_tower_vs_name, device=t).build(
                        self._input_source, self.trainer.tower_func)
                self._handles.append(self.trainer.tower_func.towers[-1])

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

    class InferencerToHookDataParallel(InferencerToHook):
        def __init__(self, inf, fetches, size):
            """
            Args:
                size(int): number of tensors to fetch per tower
            """
            super(DataParallelInferenceRunner.InferencerToHookDataParallel, self).__init__(inf, fetches)
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
        return self.InferencerToHookDataParallel(inf, fetches, sz)

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
        nr_tower = len(self._gpus)
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
