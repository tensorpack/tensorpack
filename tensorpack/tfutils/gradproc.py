#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: gradproc.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import tensorflow as tf
from abc import ABCMeta, abstractmethod
import re
from ..utils import logger, MOVING_SUMMARY_VARS_KEY

__all__ = ['GradientProcessor', 'SummaryGradient', 'CheckGradient',
           'ScaleGradient', 'MapGradient']

class GradientProcessor(object):
    __metaclass__ = ABCMeta

    def process(self, grads):
        """
        Process the symbolic gradients.

        :param grads: list of (grad, var)
        :returns: symbolic gradients with the same type as input
        """
        return self._process(grads)

    @abstractmethod
    def _process(self, grads):
        pass


_summaried_gradient = set()

class SummaryGradient(GradientProcessor):
    """
    Summary history and RMS for each graident variable
    """
    def _process(self, grads):
        for grad, var in grads:
            name = var.op.name
            if name in _summaried_gradient:
                continue
            _summaried_gradient.add(name)
            tf.histogram_summary(name + '/grad', grad)
            tf.add_to_collection(MOVING_SUMMARY_VARS_KEY,
                                 tf.sqrt(tf.reduce_mean(tf.square(grad)),
                                         name=name + '/gradRMS'))
        return grads


class CheckGradient(GradientProcessor):
    """
    Check for numeric issue
    """
    def _process(self, grads):
        for grad, var in grads:
            # TODO make assert work
            tf.Assert(tf.reduce_all(tf.is_finite(var)), [var])
        return grads

class ScaleGradient(GradientProcessor):
    """
    Scale gradient by a multiplier
    """
    def __init__(self, multipliers):
        """
        :param multipliers: list of (regex, float)
        """
        self.multipliers = multipliers

    def _process(self, grads):
        ret = []
        for grad, var in grads:
            varname = var.op.name
            for regex, val in self.multipliers:
                # always match against the whole name
                if not regex.endswith('$'):
                    regex = regex + '$'

                if re.match(regex, varname):
                    logger.info("Apply lr multiplier {} for {}".format(val, varname))
                    if val != 0:    # skip zero to speed up
                        ret.append((grad * val, var))
                    break
            else:
                ret.append((grad, var))
        return ret

class MapGradient(GradientProcessor):
    """
    Apply a function on all gradient if the name matches regex.
    """
    def __init__(self, func, regex='.*'):
        """
        :param func: takes a tensor and returns a tensor
        :param regex: used to match variables. default to match all variables.
        """
        self.func = func
        if not regex.endswith('$'):
            regex = regex + '$'
        self.regex = regex

    def _process(self, grads):
        ret = []
        for grad, var in grads:
            if re.match(self.regex, var.op.name):
                ret.append((self.func(grad), var))
            else:
                ret.append((grad, var))
        return ret
