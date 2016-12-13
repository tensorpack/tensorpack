# -*- coding: UTF-8 -*-
# File: inference.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import tensorflow as tf
import numpy as np
from abc import ABCMeta, abstractmethod
import sys
import six
from six.moves import zip

from ..utils import logger, execute_only_once
from ..utils.stats import RatioCounter, BinaryStatistics
from ..tfutils import get_op_var_name

__all__ = ['ClassificationError',
        'ScalarStats', 'Inferencer', 'BinaryClassificationStats']

@six.add_metaclass(ABCMeta)
class Inferencer(object):

    def before_inference(self):
        """
        Called before a new round of inference starts.
        """
        self._before_inference()

    def _before_inference(self):
        pass

    def datapoint(self, output):
        """
        Called after complete running every data point
        """
        self._datapoint(output)

    @abstractmethod
    def _datapoint(self, output):
        pass

    def after_inference(self):
        """
        Called after a round of inference ends.
        Returns a dict of statistics which will be logged by the InferenceRunner.
        The inferencer needs to handle other kind of logging by their own.
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

class ScalarStats(Inferencer):
    """
    Write some scalar tensor to both stat and summary.
    The output of the given Ops must be a scalar.
    The value will be averaged over all data points in the inference dataflow.
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

    def _datapoint(self, output):
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
    Compute classification error in batch mode, from a `wrong` variable

    The `wrong` tensor is supposed to be an 0/1 integer vector containing
    whether each sample in the batch is incorrectly classified.
    You can use `tf.nn.in_top_k` to produce this vector record top-k error as well.

    This callback produce the "true" error,
    taking account of the fact that batches might not have the same size in
    testing (because the size of test set might not be a multiple of batch size).
    Therefore the result is different from averaging the error rate of each batch.
    """
    def __init__(self, wrong_var_name='incorrect_vector', summary_name='val_error'):
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

    def _datapoint(self, outputs):
        vec = outputs[0]
        if vec.ndim == 0:
            logger.error("[DEPRECATED] use a 'wrong vector' for ClassificationError instead of nr_wrong")
            sys.exit(1)
        else:
            # TODO put shape assertion into inferencerrunner
            assert vec.ndim == 1, "{} is not a vector!".format(self.wrong_var_name)
            batch_size = len(vec)
            wrong = np.sum(vec)
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

    def _datapoint(self, outputs):
        pred, label = outputs
        self.stat.feed(pred, label)

    def _after_inference(self):
        return {self.prefix + '_precision': self.stat.precision,
                self.prefix + '_recall': self.stat.recall}
