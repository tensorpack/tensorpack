# -*- coding: utf-8 -*-
# File: gradproc.py


import inspect
import re
from abc import ABCMeta, abstractmethod
import six
import tensorflow as tf

from ..compat import tfv1
from ..utils import logger
from .summary import add_moving_summary
from .symbolic_functions import print_stat, rms

__all__ = ['GradientProcessor',
           'FilterNoneGrad', 'GlobalNormClip', 'MapGradient', 'SummaryGradient',
           'PrintGradient', 'CheckGradient', 'ScaleGradient']


@six.add_metaclass(ABCMeta)
class GradientProcessor(object):
    """
    Base class for all gradient processors.
    Gradient processors can be applied to optimizers by
    :func:`optimizer.apply_grad_processors`.

    Subclass should override the ``_process()`` method.
    """
    _name_scope = None

    def process(self, grads):
        """
        Process the symbolic gradients.

        Args:
            grads (list): list of (grad, var).
        Returns:
            list: processed gradients, with the same type as input.
        """

        # reuse the old name_scope, if process() is called multiple times
        if self._name_scope is None:
            with tfv1.name_scope(type(self).__name__) as scope:
                self._name_scope = scope
                return self._process(grads)
        else:
            with tfv1.name_scope(self._name_scope):
                return self._process(grads)

    @abstractmethod
    def _process(self, grads):
        pass


class FilterNoneGrad(GradientProcessor):
    """
    Skip the update and print a warning (instead of crashing),
    when the gradient of certain variable is None.
    """
    def __init__(self, verbose=True):
        """
        Args:
            verbose (bool): whether to print warning about None gradients.
        """
        super(FilterNoneGrad, self).__init__()
        self._verbose = verbose

    def _process(self, grads):
        g = []
        to_print = []
        for grad, var in grads:
            if grad is None:
                to_print.append(var.op.name)
            else:
                g.append((grad, var))
        if self._verbose and len(to_print):
            message = ', '.join(to_print)
            logger.warn("No gradient w.r.t {} trainable variables: {}".format(len(to_print), message))
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
        super(GlobalNormClip, self).__init__()
        self._norm = float(global_norm)

    def _process(self, grads):
        g = [k[0] for k in grads]
        v = [k[1] for k in grads]
        g, _ = tf.clip_by_global_norm(g, self._norm, name='clip_by_global_norm')
        return list(zip(g, v))


class MapGradient(GradientProcessor):
    """
    Apply a function on all gradient if the name matches regex.
    Keep the other gradients unchanged.

    It can be used for gradient clipping, etc.
    """

    def __init__(self, func, regex='.*'):
        """
        Args:
            func: a user-supplied function which takes one or two arguments.
                The argument(s) can be either a `grad` tensor, or `grad` and `var`.
                The function should return the new gradient to be used.
                If it return None, the gradient is discarded (hence no update to the variable will happen).
            regex (str): used to match variables. Defaults to match all variables.
        """
        args = inspect.getfullargspec(func).args
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
        super(MapGradient, self).__init__()

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


# TODO has dependency problems: sess.run may not depend on grad
# maybe group maintain op and grad ?
class SummaryGradient(MapGradient):
    """
    For each gradient tensor, summary its histogram and add it to moving
    summaries.
    """
    # avoid duplicate summaries from towers
    # TODO this is global. not good.
    _summaried_gradient = set()

    def __init__(self, regex='.*', collections=None):
        """
        Args:
            regex(str): same as in :class:`MapGradient`.
            collections (list[str]): list of collection names
        """
        super(SummaryGradient, self).__init__(self._mapper, regex)
        self._coll = collections

    def _mapper(self, grad, var):
        name = var.op.name
        if re.match('tower[0-9]+/', name):
            # replicated training, var may come from different towers
            return grad
        if name not in SummaryGradient._summaried_gradient:
            SummaryGradient._summaried_gradient.add(name)
            tfv1.summary.histogram(name + '-grad', grad, collections=self._coll)
            add_moving_summary(rms(grad, name=name + '/rms'))
        return grad


class PrintGradient(MapGradient):
    """
    Print the gradients every step with :func:`symbolic_functions.print_stat`.
    """
    _printed = set()
    # TODO this is global. not good.

    def __init__(self, regex='.*'):
        """
        Args:
            regex(str): same as in :class:`MapGradient`.
        """
        super(PrintGradient, self).__init__(self._mapper, regex)

    def _mapper(self, grad, var):
        name = var.op.name
        if name not in PrintGradient._printed:
            PrintGradient._printed.add(name)
            grad = print_stat(grad, message=name + '-grad')
        return grad


class CheckGradient(MapGradient):
    """
    Run :func:`tf.check_numerics` for each gradient.
    """

    def __init__(self):
        super(CheckGradient, self).__init__(self._mapper)

    def _mapper(self, grad, var):
        # this was very slow.... see #3649
        # op = tf.Assert(tf.reduce_all(tf.is_finite(var)), [var], summarize=100)
        grad = tf.check_numerics(grad, 'CheckGradient/' + var.op.name)
        return grad


class ScaleGradient(MapGradient):
    """
    Scale certain gradient by a multiplier.
    """

    def __init__(self, multipliers, verbose=True):
        """
        Args:
            multipliers (tuple or list): tuple of (regex, float), or list of such tuples.
            verbose (bool): whether to print logs or not

        Example:
            Use double learning rate for all the bias (as in caffe), and freeze layer0:

            .. code-block:: python

                from tensorpack.tfutils import optimizer, gradproc
                opt = optimizer.apply_grad_processors(
                    opt, [gradproc.ScaleGradient(
                        [('.*/b', 2.), ('layer0/.*', 0.)]
                    )])
        """
        if not isinstance(multipliers, list):
            multipliers = [multipliers]
        self.multipliers = multipliers
        assert verbose in [True, False], verbose
        self._verbose = verbose
        super(ScaleGradient, self).__init__(self._mapper)

    def _mapper(self, grad, var):
        varname = var.op.name
        for regex, val in self.multipliers:
            # always match against the whole name
            if not regex.endswith('$'):
                regex = regex + '$'

            if re.match(regex, varname):
                if self._verbose:
                    logger.info("Gradient of '{}' is multipled by {}".format(varname, val))
                if val != 0:    # skip zero to speed up
                    return grad * val
                else:
                    return None
        return grad
