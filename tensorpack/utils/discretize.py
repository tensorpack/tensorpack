#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: discretize.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

from . import logger, memoized
from abc import abstractmethod, ABCMeta
import numpy as np
from six.moves import range

__all__ = ['UniformDiscretizer1D']

@memoized
def log_once(s):
    logger.warn(s)

# just placeholder
class Discretizer(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def get_nr_bin(self):
        pass

    @abstractmethod
    def get_bin(self, v):
        pass

class Discretizer1D(Discretizer):
    pass

class UniformDiscretizer1D(Discretizer1D):
    def __init__(self, minv, maxv, spacing):
        """
        :params minv: minimum value of the first bin
        :params maxv: maximum value of the last bin
        :param spacing: width of a bin
        """
        self.minv = float(minv)
        self.maxv = float(maxv)
        self.spacing = float(spacing)
        self.nr_bin = int(np.ceil((self.maxv - self.minv) / self.spacing))

    def get_nr_bin(self):
        return self.nr_bin

    def get_bin(self, v):
        if v < self.minv:
            log_once("UniformDiscretizer1D: value smaller than min!")
            return 0
        if v > self.maxv:
            log_once("UniformDiscretizer1D: value larger than max!")
            return self.nr_bin - 1
        return int(np.clip(
                (v - self.minv) / self.spacing,
                0, self.nr_bin - 1))

    def get_distribution(self, v, smooth_factor=0.05, smooth_radius=2):
        """ return a smoothed one-hot distribution of the sample v.
        """
        b = self.get_bin(v)
        ret = np.zeros((self.nr_bin, ), dtype='float32')
        ret[b] = 1.0
        if v >= self.maxv or v <= self.minv:
            return ret
        try:
            for k in range(1, smooth_radius+1):
                ret[b+k] = smooth_factor ** k
        except IndexError:
            pass
        for k in range(1, min(smooth_radius+1, b+1)):
            ret[b-k] = smooth_factor ** k
        ret /= ret.sum()
        return ret


if __name__ == '__main__':
    u = UniformDiscretizer1D(-10, 10, 0.12)
    import IPython as IP;
    IP.embed(config=IP.terminal.ipapp.load_default_config())

