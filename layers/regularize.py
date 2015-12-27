#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: regularize.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import tensorflow as tf
import re

__all__ = ['regularize_cost']

def regularize_cost(regex, func):
    G = tf.get_default_graph()
    params = G.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

    cost = 0
    for p in params:
        name = p.name
        if re.search(regex, name):
            print("Weight decay for {}".format(name))
            cost += func(p)
    return cost

