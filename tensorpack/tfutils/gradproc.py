#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: gradproc.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import tensorflow as tf
from abc import ABCMeta, abstractmethod
import re
import six
import inspect
from ..utils import logger
from .symbolic_functions import rms
from .summary import add_moving_summary

__all__ = ['GradientProcessor',
           'FilterNoneGrad', 'GlobalNormClip', 'MapGradient', 'SummaryGradient',
           'CheckGradient', 'ScaleGradient']


@six.add_metaclass(ABCMeta)
class GradientProcessor(object):
    """ Base class for all gradient processors.

    Subclass should override the ``_process()`` method.
    """
    def process(self, grads):
        """
        Process the symbolic gradients.

        Args:
            grads (list): list of (grad, var).
        Returns:
            list: processed gradients, with the same type as input.
        """
        with tf.name_scope(type(self).__name__):
            return self._process(grads)

    @abstractmethod
    def _process(self, grads):
        pass


class FilterNoneGrad(GradientProcessor):
    """
    Skip the update and print a warning (instead of crashing),
    when the gradient of certain variable is None.
    """
    def _process(self, grads):
        g = []
        for grad, var in grads:
            if grad is None:
                logger.warn("No Gradient w.r.t {}".format(var.op.name))
            else:
                g.append((grad, var))
        return g


class GlobalNormClip(GradientProcessor):
    """ Clip by global norm.
        The global norm is the sum of norm for **all** gradients.

        See :func:`tf.clip_by_global_norm` for more information.
    """

    def __init__(self, global_norm):
        """
        Args:
            global_norm(float): the threshold to clip with.
        """
        self._norm = global_norm

    def _process(self, grads):
        g = [k[0] for k in grads]
        v = [k[1] for k in grads]
        g, _ = tf.clip_by_global_norm(g, self._norm, name='clip_by_global_norm')
        return list(zip(g, v))


class MapGradient(GradientProcessor):
    """
    Apply a function on all gradient if the name matches regex.
    Keep the other gradients unchanged.
    """

    def __init__(self, func, regex='.*'):
        """
        Args:
            func: takes a grad or (grad, var) pair and returns a grad. If return None, the
                gradient is discarded (hence no update to the variable will happen).
            regex (str): used to match variables. Defaults to match all variables.
        """
        args = inspect.getargspec(func).args
        arg_num = len(args) - inspect.ismethod(func)
        assert arg_num in [1, 2], \
            "The function must take 1 or 2 arguments!  ({})".format(args)
        if arg_num == 1:
            self.func = lambda grad, var: func(grad)
        else:
            self.func = func

        if not regex.endswith('$'):
            regex = regex + '$'
        self.regex = regex

    def _process(self, grads):
        ret = []
        matched = False
        for grad, var in grads:
            if re.match(self.regex, var.op.name):
                matched = True
                grad = self.func(grad, var)
                if grad is not None:
                    ret.append((grad, var))
            else:
                ret.append((grad, var))
        if not matched:
            logger.warn("[MapGradient] No match was found for regex {}.".format(self.regex))
        return ret


_summaried_gradient = set()


# TODO let the maintain op depend on grad directly ?
class SummaryGradient(MapGradient):
    """
    Summary histogram and RMS for each gradient variable.
    """

    def __init__(self):
        super(SummaryGradient, self).__init__(self._mapper)

    def _mapper(self, grad, var):
        name = var.op.name
        if name not in _summaried_gradient:
            _summaried_gradient.add(name)
            tf.summary.histogram(name + '-grad', grad)
            add_moving_summary(rms(grad, name=name + '/rms'))
        return grad


class CheckGradient(MapGradient):
    """
    Check for numeric issue.
    See :func:`tf.check_numerics` for more information.
    """

    def __init__(self):
        super(CheckGradient, self).__init__(self._mapper)

    def _mapper(self, grad, var):
        # this is very slow.... see #3649
        # op = tf.Assert(tf.reduce_all(tf.is_finite(var)), [var], summarize=100)
        grad = tf.check_numerics(grad, 'CheckGradient-' + var.op.name)
        return grad


class ScaleGradient(MapGradient):
    """
    Scale certain gradient by a multiplier.
    """

    def __init__(self, multipliers, log=True):
        """
        Args:
            multipliers (tuple or list): tuple of (regex, float), or list of tuples.
            log (bool): whether to do logging or not

        Example:
            Use double learning rate for all the bias (as in caffe):

            .. code-block:: python

                ScaleGradient(('.*/b', 2))
        """
        if not isinstance(multipliers, list):
            multipliers = [multipliers]
        self.multipliers = multipliers
        self._log = log
        super(ScaleGradient, self).__init__(self._mapper)

    def _mapper(self, grad, var):
        varname = var.op.name
        for regex, val in self.multipliers:
            # always match against the whole name
            if not regex.endswith('$'):
                regex = regex + '$'

            if re.match(regex, varname):
                if self._log:
                    logger.info("Apply lr multiplier {} for {}".format(val, varname))
                if val != 0:    # skip zero to speed up
                    return grad * val
                else:
                    return None
        return grad
