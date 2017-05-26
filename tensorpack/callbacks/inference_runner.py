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
from ..train.input_source import FeedInput, DataParallelFeedInput, FeedfreeInput
from ..predict import PredictorTowerBuilder

from .base import Callback
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
        self._input_source.setup(self.trainer.model)
        # Use predict_tower in train config. either gpuid or -1
        self._predict_tower_id = self.trainer.config.predict_tower[0]

        def fn(_):
            in_tensors = self._input_source.get_input_tensors()
            self.trainer.model.build_graph(in_tensors)
        PredictorTowerBuilder(fn, self._prefix).build(self._predict_tower_id)

        self._hooks = [self._build_hook(inf) for inf in self.infs]

    def _before_train(self):
        self._hooks.extend(self._extra_hooks)
        self._hooked_sess = HookedSession(self.trainer.sess, self._hooks)

    def _get_tensors_maybe_in_tower(self, names):
        placeholder_names = set([k.name for k in self.trainer.model.get_inputs_desc()])
        get_tensor_fn = PredictorTowerBuilder.get_tensors_maybe_in_tower
        return get_tensor_fn(placeholder_names, names, self._predict_tower_id, prefix=self._prefix)

    @abstractmethod
    def _build_hook(self, inf):
        pass

    def _trigger(self):
        for inf in self.infs:
            inf.before_inference()

        # iterate over the data, and run the hooked session
        self._input_source.reset_state()
        for _ in tqdm.trange(self._input_source.size(), **get_tqdm_kwargs()):
            feed = self._input_source.next_feed()
            self._hooked_sess.run(fetches=[], feed_dict=feed)
        summary_inferencer(self.trainer, self.infs)


class InferenceRunner(InferenceRunnerBase):
    """
    A callback that runs a list of :class:`Inferencer` on some
    :class:`DataFlow`.
    """

    def __init__(self, ds, infs, input_names=None, extra_hooks=None):
        """
        Args:
            ds (DataFlow): the DataFlow to run inferencer on.
            infs (list): a list of `Inferencer` instances.
            input_names (list[str]): list of names to match ``input``, must be a subset of the names in
                InputDesc of the model. Defaults to be all the inputs of the model.
        """
        assert isinstance(ds, DataFlow), ds
        input = FeedInput(ds, input_names)
        super(InferenceRunner, self).__init__(
            input, infs, prefix='', extra_hooks=extra_hooks)

    def _build_hook(self, inf):
        out_names = inf.get_output_tensors()
        fetches = self._get_tensors_maybe_in_tower(out_names)
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
            input_names (list[str]): same as in :class:`InferenceRunnerBase`.
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
            if name not in placeholder_names:
                ret.append(self._get_tensors_maybe_in_tower([name])[0])
            else:       # requesting an input
                idx = placeholder_names.index(name)
                ret.append(self._input_tensors[idx])
        return InferencerToHook(inf, ret)


class DataParallelInferenceRunner(InferenceRunnerBase):
    def __init__(self, ds, infs, gpus, input_names=None):
        self._tower_names = [TowerContext.get_predict_tower_name(k)
                             for k in range(len(gpus))]
        input = DataParallelFeedInput(
            ds, self._tower_names, input_names=input_names)
        super(DataParallelInferenceRunner, self).__init__(input, infs)
        self._gpus = gpus

    def _setup_graph(self):
        model = self.trainer.model
        self._input_source.setup(model)

        # build graph
        def build_tower(k):
            # inputs (placeholders) for this tower only
            input_tensors = self._input_source.get_input_tensors()
            model.build_graph(input_tensors)

        builder = PredictorTowerBuilder(build_tower, prefix=self._prefix)
        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
            for t in self._gpus:
                builder.build(t)

        # setup feeds and hooks
        self._hooks_parallel = [self._build_hook_parallel(inf) for inf in self.infs]
        self._hooks = [self._build_hook(inf) for inf in self.infs]

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
        total = self._input_source.size()
        nr_tower = len(self._gpus)
        with tqdm.tqdm(total=total, **get_tqdm_kwargs()) as pbar:
            while total >= nr_tower:
                feed = self._input_source.next_feed()
                self._parallel_hooked_sess.run(fetches=[], feed_dict=feed)
                pbar.update(nr_tower)
                total -= nr_tower
            # take care of the rest
            while total > 0:
                feed = self._input_source.next_feed(cnt=1)
                self._hooked_sess.run(fetches=[], feed_dict=feed)
                pbar.update(1)
                total -= 1
        summary_inferencer(self.trainer, self.infs)
