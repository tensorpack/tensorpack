#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: base.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>


from abc import abstractmethod, ABCMeta

__all__ = ['DataFlow', 'ProxyDataFlow']

class DataFlow(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def get_data(self):
        """
        A generator to generate data as tuple.
        """

    def size(self):
        """
        Size of this data flow.
        """
        raise NotImplementedError()

    def reset_state(self):
        """
        Reset state of the dataflow (usually the random seed)
        """
        pass

class ProxyDataFlow(DataFlow):
    def __init__(self, ds):
        self.ds = ds

    def reset_state(self):
        self.ds.reset_state()

    def size(self):
        return self.ds.size()
