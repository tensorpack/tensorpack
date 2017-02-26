# -*- coding: UTF-8 -*-
# File: inference.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import numpy as np
from abc import ABCMeta, abstractmethod
import sys
import six
from six.moves import zip

from ..utils import logger
from ..utils.stats import RatioCounter, BinaryStatistics
from ..tfutils import get_op_tensor_name

__all__ = ['ScalarStats', 'Inferencer',
           'ClassificationError', 'BinaryClassificationStats']


@six.add_metaclass(ABCMeta)
class Inferencer(object):
    """ Base class of Inferencer. To be used with :class:`InferenceRunner`. """

    def before_inference(self):
        """
        Called before a new round of inference starts.
        """
        self._before_inference()

    def _before_inference(self):
        pass

    def datapoint(self, output):
        """
        Called after each new datapoint finished the forward inference.

        Args:
            output(list): list of output this inferencer needs. Has the same
                length as ``self.get_output_tensors()``.
        """
        self._datapoint(output)

    @abstractmethod
    def _datapoint(self, output):
        pass

    def after_inference(self):
        """
        Called after a round of inference ends.
        Returns a dict of scalar statistics which will be logged to monitors.
        """
        return self._after_inference()

    def _after_inference(self):
        pass

    def get_output_tensors(self):
        """
        Return a list of tensor names (guaranteed not op name) this inferencer needs.
        """
        ret = self._get_output_tensors()
        return [get_op_tensor_name(n)[1] for n in ret]

    @abstractmethod
    def _get_output_tensors(self):
        pass


class ScalarStats(Inferencer):
    """
    Statistics of some scalar tensor.
    The value will be averaged over all given datapoints.
    """

    def __init__(self, names, prefix='validation'):
        """
        Args:
            names(list or str): list of names or just one name. The
                corresponding tensors have to be scalar.
            prefix(str): a prefix for logging
        """
        if not isinstance(names, list):
            self.names = [names]
        else:
            self.names = names
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
            opname, _ = get_op_tensor_name(name)
            name = '{}_{}'.format(self.prefix, opname) if self.prefix else opname
            ret[name] = stat
        return ret


class ClassificationError(Inferencer):
    """
    Compute classification error in batch mode, from a ``wrong`` tensor.

    The ``wrong`` tensor is supposed to be an binary vector containing
    whether each sample in the batch is *incorrectly* classified.
    You can use ``tf.nn.in_top_k`` to produce this vector.

    This Inferencer produces the "true" error,
    taking account of the fact that batches might not have the same size in
    testing (because the size of test set might not be a multiple of batch size).
    Therefore the result can be different from averaging the error rate of each batch.
    """

    def __init__(self, wrong_tensor_name='incorrect_vector', summary_name='val_error'):
        """
        Args:
            wrong_tensor_name(str): name of the ``wrong`` tensor.
                The default is the same as the default output name of
                :meth:`prediction_incorrect`.
            summary_name(str): the name to log the error with.
        """
        self.wrong_tensor_name = wrong_tensor_name
        self.summary_name = summary_name

    def _get_output_tensors(self):
        return [self.wrong_tensor_name]

    def _before_inference(self):
        self.err_stat = RatioCounter()

    def _datapoint(self, outputs):
        vec = outputs[0]
        if vec.ndim == 0:
            logger.error("[DEPRECATED] use a 'wrong vector' for ClassificationError instead of nr_wrong. Exiting..")
            sys.exit(1)
        else:
            # TODO put shape assertion into inference-runner
            assert vec.ndim == 1, "{} is not a vector!".format(self.wrong_tensor_name)
            batch_size = len(vec)
            wrong = np.sum(vec)
        self.err_stat.feed(wrong, batch_size)

    def _after_inference(self):
        return {self.summary_name: self.err_stat.ratio}


class BinaryClassificationStats(Inferencer):
    """
    Compute precision / recall in binary classification, given the
    prediction vector and the label vector.
    """

    def __init__(self, pred_tensor_name, label_tensor_name, prefix='val'):
        """
        Args:
            pred_tensor_name(str): name of the 0/1 prediction tensor.
            label_tensor_name(str): name of the 0/1 label tensor.
        """
        self.pred_tensor_name = pred_tensor_name
        self.label_tensor_name = label_tensor_name
        self.prefix = prefix

    def _get_output_tensors(self):
        return [self.pred_tensor_name, self.label_tensor_name]

    def _before_inference(self):
        self.stat = BinaryStatistics()

    def _datapoint(self, outputs):
        pred, label = outputs
        self.stat.feed(pred, label)

    def _after_inference(self):
        return {self.prefix + '_precision': self.stat.precision,
                self.prefix + '_recall': self.stat.recall}
