# -*- coding: UTF-8 -*-
# File: inference.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import tensorflow as tf
import numpy as np
from tqdm import tqdm
from abc import ABCMeta, abstractmethod
import six
from six.moves import zip, map

from ..dataflow import DataFlow
from ..utils import *
from ..utils.stat import *
from ..tfutils import *
from ..tfutils.summary import *
from .base import Callback

__all__ = ['InferenceRunner', 'ClassificationError',
        'ScalarStats', 'Inferencer', 'BinaryClassificationStats']

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
        Returns a dict of statistics.
        """
        return self._after_inference()

    def _after_inference(self):
        pass

    def get_output_tensors(self):
        """
        Return a list of tensor names needed for this inference
        """
        return self._get_output_tensors()

    @abstractmethod
    def _get_output_tensors(self):
        pass

class InferenceRunner(Callback):
    """
    A callback that runs different kinds of inferencer.
    """

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

    def _setup_graph(self):
        self.input_vars = self.trainer.model.reuse_input_vars()
        self._find_output_tensors()
        input_names = [x.name for x in self.input_vars]
        self.pred_func = self.trainer.get_predict_func(
                input_names, self.output_tensors)

    def _find_output_tensors(self):
        self.output_tensors = []    # list of names
        self.vc_to_vars = []    # list of list of (var_name: output_idx)
        for vc in self.vcs:
            vc_vars = vc.get_output_tensors()
            def find_oid(var):
                if var in self.output_tensors:
                    return self.output_tensors.index(var)
                else:
                    self.output_tensors.append(var)
                    return len(self.output_tensors) - 1
            vc_vars = [(var, find_oid(var)) for var in vc_vars]
            self.vc_to_vars.append(vc_vars)

    def _trigger_epoch(self):
        for vc in self.vcs:
            vc.before_inference()

        sess = tf.get_default_session()
        self.ds.reset_state()
        with tqdm(total=self.ds.size(), **get_tqdm_kwargs()) as pbar:
            for dp in self.ds.get_data():
                #feed = dict(zip(self.input_vars, dp))   # TODO custom dp mapping?
                #outputs = sess.run(self.output_tensors, feed_dict=feed)
                outputs = self.pred_func(dp)
                for vc, varsmap in zip(self.vcs, self.vc_to_vars):
                    vc_output = [outputs[k[1]] for k in varsmap]
                    vc.datapoint(dp, vc_output)
                pbar.update()

        for vc in self.vcs:
            ret = vc.after_inference()
            for k, v in six.iteritems(ret):
                try:
                    v = float(v)
                except:
                    logger.warn("{} returns a non-scalar statistics!".format(type(vc).__name__))
                    continue
                self.trainer.write_scalar_summary(k, v)

class ScalarStats(Inferencer):
    """
    Write some scalar tensor to both stat and summary.
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

        ret = {}
        for stat, name in zip(self.stats, self.names):
            opname, _ = get_op_var_name(name)
            name = '{}_{}'.format(self.prefix, opname) if self.prefix else opname
            ret[name] = stat
        return ret

class ClassificationError(Inferencer):
    """
    Compute classification error from a `wrong` variable

    The `wrong` variable is supposed to be an integer equal to the number of failed samples in this batch.
    You can use `tf.nn.in_top_k` to record top-k error as well.

    This callback produce the "true" error,
    taking account of the fact that batches might not have the same size in
    testing (because the size of test set might not be a multiple of batch size).
    Therefore the result is different from averaging the error rate of each batch.
    """
    def __init__(self, wrong_var_name='wrong:0', summary_name='validation_error'):
        """
        :param wrong_var_name: name of the `wrong` variable
        :param summary_name: the name for logging
        """
        self.wrong_var_name = wrong_var_name
        self.summary_name = summary_name

    def _get_output_tensors(self):
        return [self.wrong_var_name]

    def _before_inference(self):
        self.err_stat = RatioCounter()

    def _datapoint(self, dp, outputs):
        batch_size = dp[0].shape[0]   # assume batched input
        wrong = int(outputs[0])
        self.err_stat.feed(wrong, batch_size)

    def _after_inference(self):
        return {self.summary_name: self.err_stat.ratio}

class BinaryClassificationStats(Inferencer):
    """ Compute precision/recall in binary classification, given the
    prediction vector and the label vector.
    """

    def __init__(self, pred_var_name, label_var_name, summary_prefix='val'):
        """
        :param pred_var_name: name of the 0/1 prediction tensor.
        :param label_var_name: name of the 0/1 label tensor.
        """
        self.pred_var_name = pred_var_name
        self.label_var_name = label_var_name
        self.prefix = summary_prefix

    def _get_output_tensors(self):
        return [self.pred_var_name, self.label_var_name]

    def _before_inference(self):
        self.stat = BinaryStatistics()

    def _datapoint(self, dp, outputs):
        pred, label = outputs
        self.stat.feed(pred, label)

    def _after_inference(self):
        return {self.prefix + '_precision': self.stat.precision,
                self.prefix + '_recall': self.stat.recall}
