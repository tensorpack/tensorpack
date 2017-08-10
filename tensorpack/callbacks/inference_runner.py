#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: inference_runner.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import tensorflow as tf
from tensorflow.python.training.monitored_session \
    import _HookedSession as HookedSession

import itertools
from contextlib import contextmanager
import tqdm
from six.moves import range

from ..utils import logger
from ..utils.utils import get_tqdm_kwargs
from ..utils.develop import deprecated
from ..dataflow.base import DataFlow

from ..graph_builder.input_source_base import InputSource
from ..graph_builder.input_source import (
    FeedInput, QueueInput)

from .base import Callback
from .group import Callbacks
from .inference import Inferencer

__all__ = ['InferenceRunner', 'FeedfreeInferenceRunner',
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
    except (StopIteration,
            tf.errors.CancelledError,
            tf.errors.OutOfRangeError):
        logger.error(
            "[InferenceRunner] input stopped before reaching its size()! " + msg)
        raise


class InferenceRunnerBase(Callback):
    """ Base class for inference runner.
        Please note that InferenceRunner will use `input.size()` to determine
        how much iterations to run, so you want it to be accurate.
    """
    def __init__(self, input, infs, extra_hooks=None):
        """
        Args:
            input (InputSource): the input to use. Must have ``size()``.
            infs (list[Inferencer]): list of :class:`Inferencer` to run.
            extra_hooks (list[SessionRunHook]): extra :class:`SessionRunHook` to run with the evaluation.
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
            raise ValueError("Input used in InferenceRunner must have a size!")
        logger.info("InferenceRunner will eval on an InputSource of size {}".format(self._size))

        if extra_hooks is None:
            extra_hooks = []
        self._extra_hooks = extra_hooks

    def _before_train(self):
        self._hooks.extend(self._extra_hooks)
        self._hooked_sess = HookedSession(self.trainer.sess, self._hooks)
        self._input_callbacks.before_train()

    def _after_train(self):
        self._input_callbacks.after_train()


class InferenceRunner(InferenceRunnerBase):
    """
    A callback that runs a list of :class:`Inferencer` on some :class:`InputSource`.
    """

    def __init__(self, input, infs, tower_name='InferenceTower', extra_hooks=None):
        """
        Args:
            input (InputSource or DataFlow): The :class:`InputSource` to run
                inference on.  If given a DataFlow, will use :class:`FeedInput`.
            infs (list): a list of :class:`Inferencer` instances.
            tower_name (str): the name scope of the tower to build. Need to set a
                different one if multiple InferenceRunner are used.
        """
        if isinstance(input, DataFlow):
            input = FeedInput(input, infinite=False)
        assert isinstance(input, InputSource), input
        self._tower_name = tower_name
        super(InferenceRunner, self).__init__(
            input, infs, extra_hooks=extra_hooks)

    def _build_hook(self, inf):
        out_names = inf.get_fetches()
        fetches = self._tower_handle.get_tensors(out_names)
        return InferencerToHook(inf, fetches)

    def _setup_graph(self):
        # Use predict_tower in train config. either gpuid or -1
        tower_id = self.trainer.config.predict_tower[0]
        device = '/gpu:{}'.format(tower_id) if tower_id >= 0 else '/cpu:0'

        input_callbacks = self._input_source.setup(self.trainer.model.get_inputs_desc())

        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
            self._tower_handle = self.trainer.predictor_factory.build(
                self._tower_name, device, self._input_source)

        self._hooks = [self._build_hook(inf) for inf in self.infs]
        # trigger_{step,epoch}, {before,after}_epoch is ignored.
        # We assume that InputSource callbacks won't use these methods
        self._input_callbacks = Callbacks(input_callbacks)
        self._hooks.extend(self._input_callbacks.get_hooks())

        for inf in self.infs:
            inf.setup_graph(self.trainer)
        self._input_callbacks.setup_graph(self.trainer)

    def _trigger(self):
        for inf in self.infs:
            inf.before_epoch()

        # iterate over the data, and run the hooked session
        self._input_source.reset_state()
        with _inference_context():
            for _ in tqdm.trange(self._size, **get_tqdm_kwargs()):
                self._hooked_sess.run(fetches=[])
        for inf in self.infs:
            inf.trigger_epoch()


@deprecated("Just use InferenceRunner since it now accepts TensorInput!", "2017-11-11")
def FeedfreeInferenceRunner(*args, **kwargs):
    return InferenceRunner(*args, **kwargs)


class DataParallelInferenceRunner(InferenceRunnerBase):
    """
    Inference with data-parallel support on multiple GPUs.
    It will build one predict tower on each GPU, and run prediction
    with a larger batch.
    """
    def __init__(self, input, infs, gpus):
        """
        Args:
            input (DataFlow or QueueInput)
            gpus (list[int]): list of GPU id
        """
        self._tower_names = ['InferenceTower{}'.format(k) for k in range(len(gpus))]
        if isinstance(input, DataFlow):
            input = QueueInput(input)
        assert isinstance(input, QueueInput), input
        super(DataParallelInferenceRunner, self).__init__(input, infs)
        self._gpus = gpus

    def _setup_graph(self):
        cbs = self._input_source.setup(self.trainer.model.get_inputs_desc())
        # build each predict tower
        self._handles = []
        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
            for idx, t in enumerate(self._gpus):
                tower_name = self._tower_names[idx]
                device = '/gpu:{}'.format(t)
                self._handles.append(
                    self.trainer.predictor_factory.build(
                        tower_name, device, self._input_source))

        # setup callbacks and hooks
        self._input_callbacks = Callbacks(cbs)

        # InputSource might have hooks which break us.
        # e.g. hooks from StagingInputWrapper will force the consumption
        # of nr_tower datapoints in every run.
        input_hooks = self._input_callbacks.get_hooks()
        self._hooks = [self._build_hook(inf) for inf in self.infs] + input_hooks
        self._hooks_parallel = [self._build_hook_parallel(inf) for inf in self.infs] + input_hooks

        for inf in self.infs:
            inf.setup_graph(self.trainer)
        self._input_callbacks.setup_graph(self.trainer)

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

        self._input_source.reset_state()
        total = self._size
        nr_tower = len(self._gpus)
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
