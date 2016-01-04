#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: utils.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>
import os

def expand_dim_if_necessary(var, dp):
    """
    Args:
        var: a tensor
        dp: a numpy array
    Return a reshaped version of dp, if that makes it match the valid dimension of var
    """
    shape = var.get_shape().as_list()
    valid_shape = [k for k in shape if k]
    if dp.shape == tuple(valid_shape):
        new_shape = [k if k else 1 for k in shape]
        dp = dp.reshape(new_shape)
    return dp


def mkdir_p(dirname):
    assert dirname is not None
    if dirname == '':
        return
    try:
        os.makedirs(dirname)
    except OSError as e:
        if e.errno != 17:
            raise e
