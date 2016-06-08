#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: serialize.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import msgpack
import msgpack_numpy
msgpack_numpy.patch()
#import dill

__all__ = ['loads', 'dumps']

def dumps(obj):
    #return dill.dumps(obj)
    return msgpack.dumps(obj, use_bin_type=True)

def loads(buf):
    #return dill.loads(buf)
    return msgpack.loads(buf)
