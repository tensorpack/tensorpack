#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: inference_runner.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import tensorflow as tf
from tensorflow.python.training.monitored_session \
    import _HookedSession as HookedSession

from abc import ABCMeta, abstractmethod
import tqdm
import six
from six.moves import range

from ..utils import logger, get_tqdm_kwargs
from ..dataflow import DataFlow
from ..tfutils.common import get_tensors_by_names
from ..tfutils.tower import TowerContext
from ..graph_builder.input_source import (
    FeedInput, DataParallelFeedInput, FeedfreeInput)
from ..predict import PredictorTowerBuilder

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
    def __init__(self, input, infs, prefix='', extra_hooks=None):
        """
        Args:
            input (InputSource): the input to use. Must have ``size()``.
            infs (list[Inferencer]): list of :class:`Inferencer` to run.
            prefix(str): an prefix used to build the tower. Must be set
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
        self._prefix = prefix

        if extra_hooks is None:
            extra_hooks = []
        self._extra_hooks = extra_hooks

    def _setup_graph(self):
        self._input_source.setup(self.trainer.model.get_inputs_desc())
        # Use predict_tower in train config. either gpuid or -1
        tower_id = self.trainer.config.predict_tower[0]
        device = '/gpu:{}'.format(tower_id) if tower_id >= 0 else '/cpu:0'
        tower_name = TowerContext.get_predict_tower_name(tower_id, prefix=self._prefix)
        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
            self._tower_handle = self.trainer.predictor_factory.build(tower_name, device, self._input_source)

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
    A callback that runs a list of :class:`Inferencer` on some
    :class:`DataFlow`.
    """

    def __init__(self, input, infs, extra_hooks=None):
        """
        Args:
            input (FeedInput or DataFlow): the FeedInput, or the DataFlow to run inferencer on.
            infs (list): a list of `Inferencer` instances.
        """
        if isinstance(input, DataFlow):
            input = FeedInput(input)
        assert isinstance(input, FeedInput), input
        super(InferenceRunner, self).__init__(
            input, infs, prefix='', extra_hooks=extra_hooks)

    def _build_hook(self, inf):
        out_names = inf.get_output_tensors()
        fetches = self._tower_handle.get_tensors(out_names)
        return InferencerToHook(inf, fetches)


class FeedfreeInferenceRunner(InferenceRunnerBase):
    """ A callback that runs a list of :class:`Inferencer` on some
    :class:`FeedfreeInput`, such as some tensor from a TensorFlow data reading
    pipeline.
    """

    def __init__(self, input, infs, prefix='', extra_hooks=None):
        """
        Args:
            input (FeedfreeInput): the input to use. Must have ``size()``.
            infs (list): list of :class:`Inferencer` to run.
            prefix(str): an prefix used to build the tower. Must be set
                differently if more than one :class:`FeedfreeInferenceRunner` are used.
        """
        assert isinstance(input, FeedfreeInput), input
        super(FeedfreeInferenceRunner, self).__init__(
            input, infs, prefix=prefix, extra_hooks=extra_hooks)

    def _build_hook(self, inf):
        out_names = inf.get_output_tensors()    # all is tensorname
        placeholder_names = [k.name + ':0' for k in self.trainer.model.get_inputs_desc()]
        ret = []
        for name in out_names:
            assert name not in placeholder_names, "Currently inferencer don't support fetching placeholders!"
            ret.append(self._tower_handle.get_tensors([name])[0])
        return InferencerToHook(inf, ret)


# TODO some scripts to test
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
        if isinstance(input, DataFlow):
            tower_names = [TowerContext.get_predict_tower_name(k) for k in range(len(gpus))]
            input = DataParallelFeedInput(input, tower_names)
        assert isinstance(input, DataParallelFeedInput), input

        super(DataParallelInferenceRunner, self).__init__(input, infs)
        self._gpus = gpus

    def _setup_graph(self):
        model = self.trainer.model
        self._input_source.setup(model.get_inputs_desc())

        # build graph
        def build_tower(k):
            # inputs (placeholders) for this tower only
            model.build_graph(self._input_source)

        builder = PredictorTowerBuilder(build_tower, prefix=self._prefix)
        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
            for t in self._gpus:
                builder.build(t)

        # setup feeds and hooks
        self._hooks_parallel = [self._build_hook_parallel(inf) for inf in self.infs]
        self._hooks = [self._build_hook(inf) for inf in self.infs]
        cbs = self._input_source.get_callbacks()
        self._hooks_parallel.extend([CallbackToHook(cb) for cb in cbs])

    def _duplicate_names_across_towers(self, names):
        ret = []
        for t in self._gpus:
            ret.extend([TowerContext.get_predict_tower_name(t, self._prefix) +
                       '/' + n for n in names])
        return ret

    class InferencerToHookDataParallel(InferencerToHook):
        def __init__(self, inf, fetches, size):
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
        out_names = self._duplicate_names_across_towers(out_names)
        fetches = get_tensors_by_names(out_names)
        return DataParallelInferenceRunner.InferencerToHookDataParallel(
            inf, fetches, sz)

    def _build_hook(self, inf):
        out_names = inf.get_output_tensors()
        names = [TowerContext.get_predict_tower_name(
            self._gpus[0], self._prefix) + '/' + n for n in out_names]
        fetches = get_tensors_by_names(names)
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
