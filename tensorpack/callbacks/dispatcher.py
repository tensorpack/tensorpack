#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: dispatcher.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>
from ..tfutils.common import get_op_tensor_name

__all__ = ['OutputTensorDispatcer']

class OutputTensorDispatcer(object):
    def __init__(self):
        self._names = []
        self._idxs = []

    def add_entry(self, names):
        v = []
        for n in names:
            tensorname = get_op_tensor_name(n)[1]
            if tensorname in self._names:
                v.append(self._names.index(tensorname))
            else:
                self._names.append(tensorname)
                v.append(len(self._names) - 1)
        self._idxs.append(v)

    def get_all_names(self):
        return self._names

    def get_idx_for_each_entry(self):
        return self._idxs
