#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: base.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>


from abc import abstractmethod, ABCMeta
from ..utils import get_rng

__all__ = ['DataFlow', 'ProxyDataFlow']

class DataFlow(object):
    """ Base class for all DataFlow """
    __metaclass__ = ABCMeta

    @abstractmethod
    def get_data(self):
        """
        A generator to generate data as a list.
        Datapoint should be a mutable list.
        Each component should be assumed immutable.
        """

    def size(self):
        """
        Size of this data flow.
        """
        raise NotImplementedError()

    def reset_state(self):
        """
        Reset state of the dataflow. Will always be called before consuming data points.
        for example, RNG **HAS** to be reset here if used in the DataFlow.
        Otherwise it may not work well with prefetching, because different
        processes will have the same RNG state.
        """
        pass


class RNGDataFlow(DataFlow):
    """ A dataflow with rng"""
    def reset_state(self):
        self.rng = get_rng(self)

class ProxyDataFlow(DataFlow):
    """ Base class for DataFlow that proxies another"""
    def __init__(self, ds):
        """
        :param ds: a :mod:`DataFlow` instance to proxy
        """
        self.ds = ds

    def reset_state(self):
        """
        Will reset state of the proxied DataFlow
        """
        self.ds.reset_state()

    def size(self):
        return self.ds.size()
