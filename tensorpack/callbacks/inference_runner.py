#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: inference_runner.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import tensorflow as tf
from collections import namedtuple
import tqdm
import six
import copy
from six.moves import zip

from ..utils import logger, get_tqdm_kwargs
from ..dataflow import DataFlow
from ..tfutils.common import get_op_tensor_name
from ..train.input_data import TensorInput, FeedInput
from ..predict import PredictorTowerBuilder, OnlinePredictor

from .base import Triggerable
from .inference import Inferencer

__all__ = ['InferenceRunner', 'FeedfreeInferenceRunner']


class OutputTensorDispatcher(object):
    def __init__(self):
        self._names = []
        self._idxs = []
        # each element in idxs is a list
        # len(idxs) == len(inferencer)
        # the list contains the indices into names

    def add_entry(self, names):
        v = []
        for n in names:
            tensorname = get_op_tensor_name(n)[1]
            if tensorname in self._names:
                v.append(self._names.index(tensorname))
            else:
                self._names.append(tensorname)
                v.append(len(self._names) - 1)
        self._idxs.append(v)

    def get_all_names(self):
        return self._names

    def get_idx_for_each_entry(self):
        return self._idxs

    def get_names_for_each_entry(self):
        ret = []
        for t in self._idxs:
            ret.append([self._names[k] for k in t])
        return ret


def summary_inferencer(trainer, infs):
    for inf in infs:
        ret = inf.after_inference()
        for k, v in six.iteritems(ret):
            try:
                v = float(v)
            except:
                logger.warn("{} returns a non-scalar statistics!".format(type(inf).__name__))
                continue
            trainer.monitors.put(k, v)


class InferenceRunner(Triggerable):
    """
    A callback that runs a list of :class:`Inferencer` on some
    :class:`DataFlow`.
    """

    _IOTensor = namedtuple('IOTensor', ['index', 'isOutput'])

    def __init__(self, ds, infs, input_names=None):
        """
        Args:
            ds (DataFlow): the DataFlow to run inferencer on.
            infs (list): a list of `Inferencer` instances.
            input_names(list): list of tensors to feed the dataflow to.
                Defaults to all the input placeholders.
        """
        if isinstance(ds, DataFlow):
            self._input_data = FeedInput(ds)
        assert isinstance(self._input_data, FeedInput), self._input_data
        if not isinstance(infs, list):
            self.infs = [infs]
        else:
            self.infs = infs
        for v in self.infs:
            assert isinstance(v, Inferencer), v
        self.input_names = input_names  # names actually
        self._prefix = ''

    def _setup_input_names(self):
        # just use all the placeholders, if input_name is None
        if self.input_names is None:
            inputs = self.trainer.model.get_reused_placehdrs()
            self.input_names = [x.name for x in inputs]

            # TODO sparse. even if it works here, sparse still is unavailable
            # because get_tensor_by_name doesn't work for sparse

            # def get_name(x):
            #     if isinstance(x, tf.SparseTensor):
            #         return x.op.name.split('/')[0]
            #     return x.name

    def _setup_output_names(self):
        dispatcher = OutputTensorDispatcher()
        for inf in self.infs:
            dispatcher.add_entry(inf.get_output_tensors())
        all_names = dispatcher.get_all_names()

        # output names can be input placeholders, use IOTensor
        self.output_names = list(filter(
            lambda x: x not in self.input_names, all_names))
        IOTensor = InferenceRunner._IOTensor

        def find_tensors(names):
            ret = []
            for name in names:
                if name in self.input_names:
                    ret.append(IOTensor(self.input_names.index(name), False))
                else:
                    ret.append(IOTensor(self.output_names.index(name), True))
            return ret
        self.inf_to_tensors = [find_tensors(t) for t in dispatcher.get_names_for_each_entry()]
        # list of list of IOTensor

    def _setup_graph(self):
        self._input_data.setup(self.trainer.model)
        self._setup_input_names()
        # set self.output_names from inferencers, as well as the name dispatcher
        self._setup_output_names()

        in_tensors = self._find_input_tensors()

        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
            def fn(_):
                self.trainer.model.build_graph(in_tensors)
            PredictorTowerBuilder(fn, self._prefix).build(0)

        feed_tensors = self._find_feed_tensors()
        out_tensors = self._find_output_tensors()
        self.predictor = OnlinePredictor(feed_tensors, out_tensors)

    def _find_input_tensors(self):
        return self.trainer.model.get_reused_placehdrs()

    def _find_feed_tensors(self):
        placeholder_names = set([k.name for k in self.trainer.model.get_inputs_desc()])
        get_tensor_fn = PredictorTowerBuilder.get_tensors_maybe_in_tower
        return get_tensor_fn(placeholder_names, self.input_names, 0, prefix=self._prefix)

    def _find_output_tensors(self):
        placeholder_names = set([k.name for k in self.trainer.model.get_inputs_desc()])
        get_tensor_fn = PredictorTowerBuilder.get_tensors_maybe_in_tower
        return get_tensor_fn(placeholder_names, self.output_names, 0, prefix=self._prefix)

    def _trigger(self):
        for inf in self.infs:
            inf.before_inference()

        self._input_data.reset_state()
        for _ in tqdm.trange(self._input_data.size(), **get_tqdm_kwargs()):
            dp = self._input_data.next_feed()
            outputs = self.predictor(dp)
            for inf, tensormap in zip(self.infs, self.inf_to_tensors):
                inf_output = [(outputs if k.isOutput else dp)[k.index]
                              for k in tensormap]
                inf.datapoint(inf_output)
        self._write_summary_after_inference()

    def _write_summary_after_inference(self):
        summary_inferencer(self.trainer, self.infs)


class FeedfreeInferenceRunner(InferenceRunner):
    """ A callback that runs a list of :class:`Inferencer` on some
    :class:`TensorInput`, such as some tensor from a TensorFlow data reading
    pipeline.
    """

    def __init__(self, input, infs, input_names=None, prefix=''):
        """
        Args:
            input (TensorInput): the input to use. Must have ``size()``.
            infs (list): list of :class:`Inferencer` to run.
            input_names (list): must be a subset of the names in InputDesc.
            prefix(str): an prefix used to build the tower. Must be set
                differently if more than one :class:`FeedfreeInferenceRunner` are used.
        """
        assert isinstance(input, TensorInput), input
        self._input_data = input
        if not isinstance(infs, list):
            self.infs = [infs]
        else:
            self.infs = infs
        for v in self.infs:
            assert isinstance(v, Inferencer), v
        if input_names is not None:
            assert isinstance(input_names, list)
        self.input_names = input_names

        try:
            self._size = input.size()
        except NotImplementedError:
            raise ValueError("Input used in FeedfreeInferencecRunner must have a size!")
        self._prefix = prefix

    def _setup_input_names(self):
        super(FeedfreeInferenceRunner, self)._setup_input_names()
        placeholder_names = set([k.name for k in self.trainer.model.get_inputs_desc()])
        for n in self.input_names:
            opname = get_op_tensor_name(n)[0]
            assert opname in placeholder_names, \
                "[FeedfreeInferenceRunner] name {} is not a model input!".format(n)

    def _setup_output_names(self):
        dispatcher = OutputTensorDispatcher()
        for inf in self.infs:
            dispatcher.add_entry(inf.get_output_tensors())
        self.output_names = dispatcher.get_all_names()
        # TODO check names. doesn't support output an input tensor (but can support)

        IOTensor = InferenceRunner._IOTensor

        def find_tensors(names):
            return [IOTensor(self.output_names.index(n), True) for n in names]
        self.inf_to_tensors = [find_tensors(t) for t in dispatcher.get_names_for_each_entry()]

    def _find_feed_tensors(self):
        return []

    def _find_input_tensors(self):
        tensors = self._input_data.get_input_tensors()

        assert len(self.input_names) == len(tensors), \
            "[FeedfreeInferenceRunner] Input names must match the " \
            "length of the input data, but {} != {}".format(len(self.input_names), len(tensors))
        # use placeholders for the unused inputs, use TensorInput for the used inpupts
        ret = copy.copy(self.trainer.model.get_reused_placehdrs())
        for name, tensor in zip(self.input_names, tensors):
            tname = get_op_tensor_name(name)[1]
            for idx, hdr in enumerate(ret):
                if hdr.name == tname:
                    ret[idx] = tensor
                    break
            else:
                assert tname in set([k.name for k in ret]), tname
        return ret

    def _write_summary_after_inference(self):
        summary_inferencer(self.trainer, self.infs)
