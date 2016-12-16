#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: inference_runner.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import tensorflow as tf
from collections import namedtuple
import six
from six.moves import zip, range

from ..dataflow import DataFlow
from .base import Callback
from .inference import Inferencer
from .dispatcher import OutputTensorDispatcer
from ..tfutils import get_op_tensor_name
from ..utils import logger, get_tqdm
from ..train.input_data import FeedfreeInput

__all__ = ['InferenceRunner']

def summary_inferencer(trainer, infs):
    for inf in infs:
        ret = inf.after_inference()
        for k, v in six.iteritems(ret):
            try:
                v = float(v)
            except:
                logger.warn("{} returns a non-scalar statistics!".format(type(inf).__name__))
                continue
            trainer.write_scalar_summary(k, v)

class InferenceRunner(Callback):
    """
    A callback that runs different kinds of inferencer.
    """

    IOTensor = namedtuple('IOTensor', ['index', 'isOutput'])

    def __init__(self, ds, infs, input_tensors=None):
        """
        :param ds: inference dataset. a `DataFlow` instance.
        :param infs: a list of `Inferencer` instance.
        :param input_tensor_names: list of tensors to feed the dataflow to.
            default to all the input placeholders.
        """
        assert isinstance(ds, DataFlow), ds
        self.ds = ds
        if not isinstance(infs, list):
            self.infs = [infs]
        else:
            self.infs = infs
        for v in self.infs:
            assert isinstance(v, Inferencer), v
        self.input_tensors = input_tensors

    def _setup_graph(self):
        self._find_input_tensors() # these are all tensor names
        self._find_output_tensors() # may be either tensor name or op name
        self.pred_func = self.trainer.get_predict_func(
                self.input_tensors, self.output_tensors)

    def _find_input_tensors(self):
        if self.input_tensors is None:
            input_vars = self.trainer.model.get_reuse_placehdrs()
            # TODO even if it works here, sparse still is unavailable
            # because get_tensor_by_name doesn't work for sparse
            def get_name(x):
                if isinstance(x, tf.SparseTensor):
                    return x.op.name.split('/')[0]
                return x.name
            self.input_tensors = [get_name(x) for x in input_vars]

    def _find_output_tensors(self):
        dispatcer = OutputTensorDispatcer()
        for inf in self.infs:
            dispatcer.add_entry(inf.get_output_tensors())
        all_names = dispatcer.get_all_names()

        IOTensor = InferenceRunner.IOTensor
        self.output_tensors = list(filter(
            lambda x: x not in self.input_tensors, all_names))
        def find_oid(idxs):
            ret = []
            for idx in idxs:
                name = all_names[idx]
                if name in self.input_tensors:
                    ret.append(IOTensor(self.input_tensors.index(name), False))
                else:
                    ret.append(IOTensor(self.output_tensors.index(name), True))
            return ret
        self.inf_to_tensors = [find_oid(t) for t in dispatcer.get_idx_for_each_entry()]
        # list of list of (var_name: IOTensor)

    def _trigger_epoch(self):
        for inf in self.infs:
            inf.before_inference()

        sess = tf.get_default_session()
        self.ds.reset_state()
        with get_tqdm(total=self.ds.size()) as pbar:
            for dp in self.ds.get_data():
                outputs = self.pred_func(dp)
                for inf, tensormap in zip(self.infs, self.inf_to_tensors):
                    inf_output = [(outputs if k.isOutput else dp)[k.index]
                            for k in tensormap]
                    inf.datapoint(inf_output)
                pbar.update()
        self._write_summary_after_inference()

    def _write_summary_after_inference(self):
        summary_inferencer(self.trainer, self.infs)

class FeedfreeInferenceRunner(Callback):
    IOTensor = namedtuple('IOTensor', ['index', 'isOutput'])

    def __init__(self, input, infs, input_tensors=None):
        assert isinstance(input, FeedfreeInput), input
        self._input_data = input
        if not isinstance(infs, list):
            self.infs = [infs]
        else:
            self.infs = infs
        for v in self.infs:
            assert isinstance(v, Inferencer), v
        self.input_tensor_names = input_tensors

    def _setup_graph(self):
        self._find_input_tensors()  # tensors
        self._find_output_tensors()
        # TODO build tower

    def _find_input_tensors(self):
        self._input_data._setup(self.trainer)
        # only 1 prediction tower will be used for inference
        self._input_tensors = self._input_data.get_input_tensors()
        model_placehdrs = self.trainer.model.get_reuse_placehdrs()
        assert len(self._input_tensors) == len(model_placehdrs), \
            "FeedfreeInput doesn't produce correct number of output tensors"
        if self.input_tensor_names is not None:
            assert isinstance(self.input_tensor_names, list)
            self._input_tensors = [k for idx, k in enumerate(self._input_tensors)
                    if model_placehdrs[idx].name in self.input_tensor_names]
            assert len(self._input_tensors) == len(self.input_tensor_names), \
                    "names of input tensors are not defined in the Model"

    def _find_output_tensors(self):
        # doesn't support output an input tensor
        dispatcer = OutputTensorDispatcer()
        for inf in self.infs:
            dispatcer.add_entry(inf.get_output_tensors())
        all_names = dispatcer.get_all_names()

        IOTensor = InferenceRunner.IOTensor
        self.output_tensors = all_names
        def find_oid(idxs):
            ret = []
            for idx in idxs:
                name = all_names[idx]
                ret.append(IOTensor(self.output_tensors.index(name), True))
            return ret
        self.inf_to_tensors = [find_oid(t) for t in dispatcer.get_idx_for_each_entry()]
        # list of list of (var_name: IOTensor)


    def _trigger_epoch(self):
        for inf in self.infs:
            inf.before_inference()

        sess = tf.get_default_session()
        sz = self._input_data.size()
        with get_tqdm(total=sz) as pbar:
            for _ in range(sz):
                #outputs = self.pred_func(dp)
                #for inf, tensormap in zip(self.infs, self.inf_to_tensors):
                    #inf_output = [(outputs if k.isOutput else dp)[k.index]
                            #for k in tensormap]
                    #inf.datapoint(inf_output)
                pbar.update()
        self._write_summary_after_inference()

    def _write_summary_after_inference(self):
        summary_inferencer(self.trainer, self.infs)
