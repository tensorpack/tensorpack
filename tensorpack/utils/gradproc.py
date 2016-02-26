#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: gradproc.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import tensorflow as tf
from abc import ABCMeta, abstractmethod
import re
from . import logger

__all__ = ['GradientProcessor', 'SummaryGradient', 'CheckGradient',
           'ScaleGradient']

class GradientProcessor(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def process(self, grads):
        """
        Process the symbolic gradients, return symbolic gradients
        grads: list of (grad, var)
        """

class SummaryGradient(GradientProcessor):
    """
    Summary history and RMS for each graident variable
    """
    def process(self, grads):
        for grad, var in grads:
            tf.histogram_summary(var.op.name + '/grad', grad)
            tf.scalar_summary(var.op.name + '/gradRMS',
                              tf.sqrt(tf.reduce_mean(tf.square(grad))))
        return grads


class CheckGradient(GradientProcessor):
    """
    Check for numeric issue
    """
    def process(self, grads):
        for grad, var in grads:
            assert grad is not None, "Grad is None for variable {}".format(var.name)
            # TODO make assert work
            tf.Assert(tf.reduce_all(tf.is_finite(var)), [var])
        return grads

class ScaleGradient(GradientProcessor):
    """
    Scale gradient by a multiplier
    """
    def __init__(self, multipliers):
        """
        multipliers: list of (regex, float)
        """
        self.multipliers = multipliers

    def process(self, grads):
        # TODO use None for zero to speed up?
        ret = []
        for grad, var in grads:
            varname = var.op.name
            for regex, val in self.multipliers:
                if re.search(regex, varname):
                    logger.info("Apply lr multiplier {} for {}".format(val, varname))
                    ret.append((grad * val, var))
                    break
            else:
                ret.append((grad, var))
        return ret
