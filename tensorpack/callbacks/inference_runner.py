#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: inference_runner.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import tensorflow as tf
from tensorflow.python.training.monitored_session \
    import _HookedSession as HookedSession

import itertools
from abc import ABCMeta, abstractmethod
import tqdm
import six
from six.moves import range

from ..utils import logger
from ..utils.utils import get_tqdm_kwargs
from ..utils.develop import deprecated
from ..dataflow import DataFlow

from ..graph_builder.input_source_base import InputSource
from ..graph_builder.input_source import (
    FeedInput, DataParallelFeedInput, FeedfreeInput, TensorInput)

from .base import Callback
from .inference import Inferencer
from .hooks import CallbackToHook

__all__ = ['InferenceRunner', 'FeedfreeInferenceRunner',
           'DataParallelInferenceRunner']


class InferencerToHook(tf.train.SessionRunHook):
    def __init__(self, inf, fetches):
        self._inf = inf
        self._fetches = fetches

    def before_run(self, _):
        return tf.train.SessionRunArgs(fetches=self._fetches)

    def after_run(self, _, run_values):
        self._inf.datapoint(run_values.results)


def summary_inferencer(trainer, infs):
    for inf in infs:
        ret = inf.after_inference()
        if ret is None:
            continue
        for k, v in six.iteritems(ret):
            try:
                v = float(v)
                trainer.monitors.put_scalar(k, v)
            except:
                logger.warn("{} returns a non-scalar statistics!".format(type(inf).__name__))
                continue


@six.add_metaclass(ABCMeta)
class InferenceRunnerBase(Callback):
    """ Base methods for inference runner"""
    def __init__(self, input, infs, tower_name='InferenceTower', extra_hooks=None, prefix=None):
        """
        Args:
            input (InputSource): the input to use. Must have ``size()``.
            infs (list[Inferencer]): list of :class:`Inferencer` to run.
            tower_name(str): name scope to build the tower. Must be set
                differently if more than one :class:`InferenceRunner` are used.
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
        self._tower_name = tower_name
        if prefix is not None:
            self._tower_name = 'InferenceTower' + prefix

        if extra_hooks is None:
            extra_hooks = []
        self._extra_hooks = extra_hooks

    def _setup_graph(self):
        # Use predict_tower in train config. either gpuid or -1
        tower_id = self.trainer.config.predict_tower[0]
        device = '/gpu:{}'.format(tower_id) if tower_id >= 0 else '/cpu:0'

        self._input_source.setup(self.trainer.model.get_inputs_desc())
        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
            self._tower_handle = self.trainer.predictor_factory.build(self._tower_name, device, self._input_source)

        self._hooks = [self._build_hook(inf) for inf in self.infs]
        cbs = self._input_source.get_callbacks()
        self._hooks.extend([CallbackToHook(cb) for cb in cbs])

    def _before_train(self):
        self._hooks.extend(self._extra_hooks)
        self._hooked_sess = HookedSession(self.trainer.sess, self._hooks)

    @abstractmethod
    def _build_hook(self, inf):
        pass

    def _trigger(self):
        for inf in self.infs:
            inf.before_inference()

        # iterate over the data, and run the hooked session
        self._input_source.reset_state()
        for _ in tqdm.trange(self._size, **get_tqdm_kwargs()):
            self._hooked_sess.run(fetches=[])
        summary_inferencer(self.trainer, self.infs)


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
        """
        if isinstance(input, DataFlow):
            input = FeedInput(input)
        assert isinstance(input, InputSource), input
        if isinstance(input, FeedfreeInput):    # TODO support other input
            assert isinstance(input, TensorInput), "InferenceRunner only accepts TensorInput or FeedInput!"
        super(InferenceRunner, self).__init__(
            input, infs, tower_name=tower_name, extra_hooks=extra_hooks)

    def _build_hook(self, inf):
        out_names = inf.get_output_tensors()
        fetches = self._tower_handle.get_tensors(out_names)
        return InferencerToHook(inf, fetches)


@deprecated("Just use InferenceRunner since it now accepts TensorInput!")
def FeedfreeInferenceRunner(*args, **kwargs):
    return InferenceRunner(*args, **kwargs)


class DataParallelInferenceRunner(InferenceRunnerBase):
    """
    Inference by feeding datapoints in a data-parallel way to multiple GPUs.

    Doesn't support remapped InputSource for now.
    """
    def __init__(self, input, infs, gpus):
        """
        Args:
            input (DataParallelFeedInput or DataFlow)
            gpus (list[int]): list of GPU id
        """
        self._tower_names = ['InferenceTower{}'.format(k) for k in range(len(gpus))]
        if isinstance(input, DataFlow):
            input = DataParallelFeedInput(input, self._tower_names)
        assert isinstance(input, DataParallelFeedInput), input

        super(DataParallelInferenceRunner, self).__init__(input, infs)
        self._gpus = gpus

    def _setup_graph(self):
        self._input_source.setup(self.trainer.model.get_inputs_desc())
        self._handles = []
        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
            for idx, t in enumerate(self._gpus):
                tower_name = self._tower_names[idx]
                device = '/gpu:{}'.format(t)
                self._handles.append(
                    self.trainer.predictor_factory.build(
                        tower_name, device, self._input_source))

        # setup feeds and hooks
        self._hooks_parallel = [self._build_hook_parallel(inf) for inf in self.infs]
        self._hooks = [self._build_hook(inf) for inf in self.infs]
        cbs = self._input_source.get_callbacks()
        self._hooks_parallel.extend([CallbackToHook(cb) for cb in cbs])

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
                self._inf.datapoint(vals)

    def _build_hook_parallel(self, inf):
        out_names = inf.get_output_tensors()
        sz = len(out_names)
        fetches = list(itertools.chain(*[t.get_tensors(out_names) for t in self._handles]))
        return self.InferencerToHookDataParallel(inf, fetches, sz)

    def _build_hook(self, inf):
        out_names = inf.get_output_tensors()
        fetches = self._handles[0].get_tensors(out_names)
        return InferencerToHook(inf, fetches)

    def _before_train(self):
        self._hooked_sess = HookedSession(self.trainer.sess, self._hooks)
        self._parallel_hooked_sess = HookedSession(self.trainer.sess, self._hooks_parallel)

    def _trigger(self):
        for inf in self.infs:
            inf.before_inference()

        self._input_source.reset_state()
        total = self._size
        nr_tower = len(self._gpus)
        with tqdm.tqdm(total=total, **get_tqdm_kwargs()) as pbar:
            while total >= nr_tower:
                self._parallel_hooked_sess.run(fetches=[])
                pbar.update(nr_tower)
                total -= nr_tower
            # take care of the rest
            try:
                while total > 0:
                    # TODO XXX doesn't support remap
                    feed = self._input_source.next_feed(cnt=1)
                    self._hooked_sess.run(fetches=[], feed_dict=feed)
                    pbar.update(1)
                    total -= 1
            except AttributeError:
                logger.error(
                    "[DataParallelInferenceRunner] doesn't support InputSource wrappers very well!")
                logger.error("[DataParallelInferenceRunner] Skipping the rest of the datapoints ...")
        summary_inferencer(self.trainer, self.infs)
