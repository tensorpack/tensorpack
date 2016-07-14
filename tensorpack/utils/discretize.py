#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: discretize.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

from . import logger, memoized
from abc import abstractmethod, ABCMeta
import numpy as np
from six.moves import range

__all__ = ['UniformDiscretizer1D', 'UniformDiscretizerND']

@memoized
def log_once(s):
    logger.warn(s)

# just a placeholder
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

    def get_bin_center(self, bin_id):
        return self.minv + self.spacing * (bin_id + 0.5)

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


class UniformDiscretizerND(Discretizer):
    def __init__(self, *min_max_spacing):
        """
        :params min_max_spacing: (minv, maxv, spacing) for each dimension
        """
        self.n = len(min_max_spacing)
        self.discretizers = [UniformDiscretizer1D(*k) for k in min_max_spacing]
        self.nr_bins = [k.get_nr_bin() for k in self.discretizers]

    def get_nr_bin(self):
        return np.prod(self.nr_bins)

    def get_bin(self, v):
        assert len(v) == self.n
        bin_id = [self.discretizers[k].get_bin(v[k]) for k in range(self.n)]
        return self.get_bin_from_nd_bin_ids(bin_id)

    def get_nd_bin_ids(self, bin_id):
        ret = []
        for k in reversed(list(range(self.n))):
            nr = self.nr_bins[k]
            v = bin_id % nr
            bin_id = bin_id / nr
            ret.append(v)
        return list(reversed(ret))

    def get_bin_from_nd_bin_ids(self, bin_ids):
        acc, res = 1, 0
        for k in reversed(list(range(self.n))):
            res += bin_ids[k] * acc
            acc *= self.nr_bins[k]
        return res

    def get_nr_bin_nd(self):
        return self.nr_bins

    def get_bin_center(self, bin_id):
        bin_id_nd = self.get_nd_bin_ids(bin_id)
        return [self.discretizers[k].get_bin_center(bin_id_nd[k]) for k in range(self.n)]

if __name__ == '__main__':
    #u = UniformDiscretizer1D(-10, 10, 0.12)
    u = UniformDiscretizerND((0, 100, 1), (0, 100, 1), (0, 100, 1))
    import IPython as IP;
    IP.embed(config=IP.terminal.ipapp.load_default_config())

