#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: lut.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import six

__all__ = ['LookUpTable']


class LookUpTable(object):
    """ Maintain mapping from index to objects. """

    def __init__(self, objlist):
        """
        Args:
            objlist(list): list of objects
        """
        self.idx2obj = dict(enumerate(objlist))
        self.obj2idx = {v: k for k, v in six.iteritems(self.idx2obj)}

    def size(self):
        return len(self.idx2obj)

    def get_obj(self, idx):
        return self.idx2obj[idx]

    def get_idx(self, obj):
        return self.obj2idx[obj]

    def __str__(self):
        return self.idx2obj.__str__()
