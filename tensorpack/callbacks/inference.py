# -*- coding: UTF-8 -*-
# File: inference.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import tensorflow as tf
from tqdm import tqdm
from abc import ABCMeta, abstractmethod
from six.moves import zip, map

from ..dataflow import DataFlow
from ..utils import *
from ..utils.stat import *
from ..tfutils import *
from ..tfutils.summary import *
from .base import Callback, TestCallbackType

__all__ = ['InferenceRunner', 'ClassificationError',
        'ScalarStats', 'Inferencer']

class Inferencer(object):
    __metaclass__ = ABCMeta

    def before_inference(self):
        """
        Called before a new round of inference starts.
        """
        self._before_inference()

    def _before_inference(self):
        pass

    def datapoint(self, dp, output):
        """
        Called after complete running every data point
        """
        self._datapoint(dp, output)

    @abstractmethod
    def _datapoint(self, dp, output):
        pass

    def after_inference(self):
        """
        Called after a round of inference ends.
        """
        self._after_inference()

    def _after_inference(self):
        pass

    def get_output_tensors(self):
        """
        Return a list of tensor names needed for this inference
        """
        return self._get_output_vars()

    @abstractmethod
    def _get_output_tensors(self):
        pass

class InferenceRunner(Callback):
    """
    A callback that runs different kinds of inferencer.
    """
    type = TestCallbackType()

    def __init__(self, ds, vcs):
        """
        :param ds: inference dataset. a `DataFlow` instance.
        :param vcs: a list of `Inferencer` instance.
        """
        assert isinstance(ds, DataFlow), type(ds)
        self.ds = ds
        if not isinstance(vcs, list):
            self.vcs = [vcs]
        else:
            self.vcs = vcs
        for v in self.vcs:
            assert isinstance(v, Inferencer), str(v)

    def _before_train(self):
        self.input_vars = self.trainer.model.reuse_input_vars()
        self._find_output_tensors()
        for v in self.vcs:
            v.trainer = self.trainer

    def _find_output_tensors(self):
        self.output_tensors = []
        self.vc_to_vars = []
        for vc in self.vcs:
            vc_vars = vc._get_output_tensors()
            def find_oid(var):
                if var in self.output_tensors:
                    return self.output_tensors.index(var)
                else:
                    self.output_tensors.append(var)
                    return len(self.output_tensors) - 1
            vc_vars = [(var, find_oid(var)) for var in vc_vars]
            self.vc_to_vars.append(vc_vars)

        # convert name to tensors
        def get_tensor(name):
            _, varname = get_op_var_name(name)
            return self.graph.get_tensor_by_name(varname)
        self.output_tensors = list(map(get_tensor, self.output_tensors))

    def _trigger_epoch(self):
        for vc in self.vcs:
            vc.before_inference()

        sess = tf.get_default_session()
        with tqdm(total=self.ds.size(), ascii=True) as pbar:
            for dp in self.ds.get_data():
                feed = dict(zip(self.input_vars, dp))   # TODO custom dp mapping?
                outputs = sess.run(self.output_tensors, feed_dict=feed)
                for vc, varsmap in zip(self.vcs, self.vc_to_vars):
                    vc_output = [outputs[k[1]] for k in varsmap]
                    vc.datapoint(dp, vc_output)
                pbar.update()

        for vc in self.vcs:
            vc.after_inference()

class ScalarStats(Inferencer):
    """
    Write stat and summary of some scalar tensor.
    The output of the given Ops must be a scalar.
    The value will be averaged over all data points in the dataset.
    """
    def __init__(self, names_to_print, prefix='validation'):
        """
        :param names_to_print: list of names of tensors, or just a name
        :param prefix: an optional prefix for logging
        """
        if not isinstance(names_to_print, list):
            self.names = [names_to_print]
        else:
            self.names = names_to_print
        self.prefix = prefix

    def _get_output_tensors(self):
        return self.names

    def _before_inference(self):
        self.stats = []

    def _datapoint(self, dp, output):
        self.stats.append(output)

    def _after_inference(self):
        self.stats = np.mean(self.stats, axis=0)
        assert len(self.stats) == len(self.names)

        for stat, name in zip(self.stats, self.names):
            opname, _ = get_op_var_name(name)
            name = '{}_{}'.format(self.prefix, opname) if self.prefix else opname
            self.trainer.summary_writer.add_summary(
                    create_summary(name, stat), get_global_step())
            self.trainer.stat_holder.add_stat(name, stat)

class ClassificationError(Inferencer):
    """
    Validate the accuracy from a `wrong` variable

    The `wrong` variable is supposed to be an integer equal to the number of failed samples in this batch

    This callback produce the "true" error,
    taking account of the fact that batches might not have the same size in
    testing (because the size of test set might not be a multiple of batch size).
    In theory, the result could be different from what produced by ValidationStatPrinter.
    """
    def __init__(self, wrong_var_name='wrong:0', prefix='validation'):
        """
        :param wrong_var_name: name of the `wrong` variable
        :param prefix: an optional prefix for logging
        """
        self.wrong_var_name = wrong_var_name
        self.prefix = prefix

    def _get_output_tensors(self):
        return [self.wrong_var_name]

    def _before_inference(self):
        self.err_stat = Accuracy()

    def _datapoint(self, dp, outputs):
        batch_size = dp[0].shape[0]   # assume batched input
        wrong = int(outputs[0])
        self.err_stat.feed(wrong, batch_size)

    def _after_inference(self):
        self.trainer.summary_writer.add_summary(
                create_summary('{}_error'.format(self.prefix), self.err_stat.accuracy),
                get_global_step())
        self.trainer.stat_holder.add_stat("{}_error".format(self.prefix), self.err_stat.accuracy)
