# -*- coding: UTF-8 -*-
# File: stat.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>
import numpy as np

__all__ = ['StatCounter', 'Accuracy', 'BinaryStatistics']

class StatCounter(object):
    def __init__(self):
        self.reset()

    def feed(self, v):
        self.values.append(v)

    def reset(self):
        self.values = []

    @property
    def count(self):
        return len(self.values)

    @property
    def average(self):
        return np.mean(self.values)

    @property
    def sum(self):
        return np.sum(self.values)

    @property
    def max(self):
        return max(self.values)

class Accuracy(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.tot = 0
        self.corr = 0

    def feed(self, corr, tot=1):
        self.tot += tot
        self.corr += corr

    @property
    def accuracy(self):
        if self.tot == 0:
            return 0
        return self.corr * 1.0 / self.tot


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
        :param pred: 0/1 np array
        :param label: 0/1 np array of the same size
        """
        nr = label.size
        assert pred.size == label.size
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
