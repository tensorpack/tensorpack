#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: _common.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

__all__ = ['layer_register']
import tensorflow as tf

def layer_register():
    def wrapper(func):
        def inner(*args, **kwargs):
            name = args[0]
            assert isinstance(name, basestring)
            args = args[1:]

            with tf.name_scope(name):
                return func(*args, **kwargs)
        return inner
    return wrapper
