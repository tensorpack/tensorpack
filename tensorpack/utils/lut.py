#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: lut.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import six

__all__ = ['LookUpTable']

class LookUpTable(object):
    def __init__(self, objlist):
        self.idx2obj = dict(enumerate(objlist))
        self.obj2idx = {v : k for k, v in six.iteritems(self.idx2obj)}
