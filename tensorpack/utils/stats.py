# -*- coding: utf-8 -*-
# File: stats.py

import numpy as np

__all__ = ['StatCounter', 'BinaryStatistics', 'RatioCounter', 'Accuracy',
           'OnlineMoments']


class StatCounter(object):
    """ A simple counter"""

    def __init__(self):
        self.reset()

    def feed(self, v):
        """
        Args:
            v(float or np.ndarray): has to be the same shape between calls.
        """
        self._values.append(v)

    def reset(self):
        self._values = []

    @property
    def count(self):
        return len(self._values)

    @property
    def average(self):
        assert len(self._values)
        return np.mean(self._values)

    @property
    def sum(self):
        assert len(self._values)
        return np.sum(self._values)

    @property
    def max(self):
        assert len(self._values)
        return max(self._values)

    @property
    def min(self):
        assert len(self._values)
        return min(self._values)

    def samples(self):
        """
        Returns all samples.
        """
        return self._values


class RatioCounter(object):
    """ A counter to count ratio of something. """

    def __init__(self):
        self.reset()

    def reset(self):
        self._tot = 0
        self._cnt = 0

    def feed(self, count, total=1):
        """
        Args:
            cnt(int): the count of some event of interest.
            tot(int): the total number of events.
        """
        self._tot += total
        self._cnt += count

    @property
    def ratio(self):
        if self._tot == 0:
            return 0
        return self._cnt * 1.0 / self._tot

    @property
    def total(self):
        """
        Returns:
            int: the total
        """
        return self._tot

    @property
    def count(self):
        """
        Returns:
            int: the total
        """
        return self._cnt


class Accuracy(RatioCounter):
    """ A RatioCounter with a fancy name """
    @property
    def accuracy(self):
        return self.ratio


class BinaryStatistics(object):
    """
    Statistics for binary decision,
    including precision, recall, false positive, false negative
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.nr_pos = 0  # positive label
        self.nr_neg = 0  # negative label
        self.nr_pred_pos = 0
        self.nr_pred_neg = 0
        self.corr_pos = 0   # correct predict positive
        self.corr_neg = 0   # correct predict negative

    def feed(self, pred, label):
        """
        Args:
            pred (np.ndarray): binary array.
            label (np.ndarray): binary array of the same size.
        """
        assert pred.shape == label.shape, "{} != {}".format(pred.shape, label.shape)
        self.nr_pos += (label == 1).sum()
        self.nr_neg += (label == 0).sum()
        self.nr_pred_pos += (pred == 1).sum()
        self.nr_pred_neg += (pred == 0).sum()
        self.corr_pos += ((pred == 1) & (pred == label)).sum()
        self.corr_neg += ((pred == 0) & (pred == label)).sum()

    @property
    def precision(self):
        if self.nr_pred_pos == 0:
            return 0
        return self.corr_pos * 1. / self.nr_pred_pos

    @property
    def recall(self):
        if self.nr_pos == 0:
            return 0
        return self.corr_pos * 1. / self.nr_pos

    @property
    def false_positive(self):
        if self.nr_pred_pos == 0:
            return 0
        return 1 - self.precision

    @property
    def false_negative(self):
        if self.nr_pos == 0:
            return 0
        return 1 - self.recall


class OnlineMoments(object):
    """Compute 1st and 2nd moments online (to avoid storing all elements).

    See algorithm at: https://www.wikiwand.com/en/Algorithms_for_calculating_variance#/Online_algorithm
    """

    def __init__(self):
        self._mean = 0
        self._M2 = 0
        self._n = 0

    def feed(self, x):
        """
        Args:
            x (float or np.ndarray): must have the same shape.
        """
        self._n += 1
        delta = x - self._mean
        self._mean += delta * (1.0 / self._n)
        delta2 = x - self._mean
        self._M2 += delta * delta2

    @property
    def mean(self):
        return self._mean

    @property
    def variance(self):
        return self._M2 / (self._n - 1)

    @property
    def std(self):
        return np.sqrt(self.variance)
