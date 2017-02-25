#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: param.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import tensorflow as tf
from abc import abstractmethod, ABCMeta
import operator
import six
import os

from .base import Triggerable
from ..utils import logger
from ..tfutils import get_op_tensor_name

__all__ = ['HyperParam', 'GraphVarParam', 'ObjAttrParam',
           'HyperParamSetter', 'HumanHyperParamSetter',
           'ScheduledHyperParamSetter',
           'StatMonitorParamSetter', 'HyperParamSetterWithFunc',
           ]


@six.add_metaclass(ABCMeta)
class HyperParam(object):
    """ Base class for a hyperparam. """

    def setup_graph(self):
        """ setup the graph in ``setup_graph`` callback stage, if necessary"""
        pass

    @abstractmethod
    def set_value(self, v):
        """
        Set the value of the param.

        Args:
            v: the value to be set
        """
        pass

    @abstractmethod
    def get_value(self):
        """
        Get the value of the param.
        """
        pass

    @property
    def readable_name(self):
        """ A name to display """
        return self._readable_name


class GraphVarParam(HyperParam):
    """ A variable in the graph (e.g. learning_rate) can be a hyperparam"""

    def __init__(self, name, shape=[]):
        """
        Args:
            name(str): name of the variable.
            shape(list): shape of the variable.
        """
        self.name = name
        self.shape = shape
        self._readable_name, self.var_name = get_op_tensor_name(name)

    def setup_graph(self):
        """ Will setup the assign operator for that variable. """
        all_vars = tf.global_variables()
        for v in all_vars:
            if v.name == self.var_name:
                self.var = v
                break
        else:
            raise ValueError("{} is not a VARIABLE in the graph!".format(self.var_name))

    def set_value(self, v):
        """ Assign the variable a new value. """
        self.var.load(v)

    def get_value(self):
        """ Evaluate the variable. """
        return self.var.eval()


class ObjAttrParam(HyperParam):
    """ An attribute of an object can be a hyperparam. """

    def __init__(self, obj, attrname, readable_name=None):
        """
        Args:
            obj: the object
            attrname (str): the attribute
            readable_name(str): The name to display. Defaults to be ``attrname``.
        """
        self.obj = obj
        self.attrname = attrname
        if readable_name is None:
            self._readable_name = attrname
        else:
            self._readable_name = readable_name

    def set_value(self, v):
        setattr(self.obj, self.attrname, v)

    def get_value(self, v):
        return getattr(self.obj, self.attrname)


class HyperParamSetter(Triggerable):
    """
    An abstract base callback to set hyperparameters.
    """

    def __init__(self, param):
        """
        Args:
            param(HyperParam or str): if is a :class:`str`, it is assumed to
                be a :class:`GraphVarParam`.
        """
        # if a string, assumed to be a scalar graph variable
        if isinstance(param, six.string_types):
            param = GraphVarParam(param)
        assert isinstance(param, HyperParam), type(param)
        self.param = param
        self.last_value = None

    def _setup_graph(self):
        self.param.setup_graph()

    def get_value_to_set(self):
        """
        Returns:
            The value to assign to the variable.

        Note:
            Subclasses will implement the abstract method
            :meth:`_get_value_to_set`, which should return a new value to
            set, or return None to do nothing.
        """
        ret = self._get_value_to_set()
        if ret is not None and ret != self.last_value:
            logger.info("{} at epoch {} will change to {:.8f}".format(
                self.param.readable_name, self.epoch_num + 1, ret))
        self.last_value = ret
        return ret

    @abstractmethod
    def _get_value_to_set(self):
        pass

    def get_current_value(self):
        """
        Returns:
            The current value of the param.
        """
        return self.param.get_value()

    def _trigger(self):
        self._set_param()

    def _before_train(self):
        self._set_param()

    def _set_param(self):
        v = self.get_value_to_set()
        if v is not None:
            self.param.set_value(v)


class HumanHyperParamSetter(HyperParamSetter):
    """
    Set hyperparameter by loading the value from a file each time it get called.
    This is useful for manually tuning some parameters (e.g. learning_rate)
    without interrupting the training.
    """

    def __init__(self, param, file_name='hyper.txt'):
        """
        Args:
            param: same as in :class:`HyperParamSetter`.
            file_name(str): a file containing the value of the variable.
                Each line in the file is a k:v pair, where k is
                param.readable_name, and v is the value. If the pair is not found,
                the param will not be changed.
        """
        super(HumanHyperParamSetter, self).__init__(param)
        self.file_name = os.path.join(logger.LOG_DIR, file_name)
        logger.info("Use {} to control hyperparam {}.".format(
            self.file_name, self.param.readable_name))

    def _get_value_to_set(self):
        # ignore if no such file exists
        if not os.path.isfile(self.file_name):
            return None
        try:
            with open(self.file_name) as f:
                lines = f.readlines()
            lines = [s.strip().split(':') for s in lines]
            dic = {str(k): float(v) for k, v in lines}
            ret = dic[self.param.readable_name]
            return ret
        except:
            logger.warn(
                "Cannot find {} in {}".format(
                    self.param.readable_name, self.file_name))
            return None


class ScheduledHyperParamSetter(HyperParamSetter):
    """
    Set hyperparameters by a predefined epoch-based schedule.
    """

    def __init__(self, param, schedule, interp=None):
        """
        Args:
            param: same as in :class:`HyperParamSetter`.
            schedule (list): with the format ``[(epoch1, val1), (epoch2, val2), (epoch3, val3)]``.
                Each ``(ep, val)`` pair means to set the param
                to "val" __after__ the completion of `ep` th epoch.
                If ep == 0, the value will be set before the first epoch.
            interp: None: no interpolation. 'linear': linear interpolation

        Example:
            .. code-block:: python

                ScheduledHyperParamSetter('learning_rate',
                                          [(30, 1e-2), (60, 1e-3), (85, 1e-4), (95, 1e-5)]),
        """
        schedule = [(int(a), float(b)) for a, b in schedule]
        self.schedule = sorted(schedule, key=operator.itemgetter(0))
        if interp is not None:
            assert interp == 'linear'
        self.interp = interp
        super(ScheduledHyperParamSetter, self).__init__(param)

    def _get_value_to_set(self):
        if self.interp is None:
            for e, v in self.schedule:
                if e == self.epoch_num:
                    return v
            return None
        else:
            laste, lastv = None, None
            for e, v in self.schedule:
                if e == self.epoch_num:
                    return v
                if e > self.epoch_num:
                    break
                laste, lastv = e, v
            if laste is None or laste == e:
                # hasn't reached the first scheduled point, or reached the end of all scheduled points
                return None
            v = (self.epoch_num - laste) * 1. / (e - laste) * (v - lastv) + lastv
            return v


class HyperParamSetterWithFunc(HyperParamSetter):
    """ Set the parameter by a function of epoch num and old value. """
    def __init__(self, param, func):
        """
        Args:
            param: same as in :class:`HyperParamSetter`.
            func: ``param`` will be set by ``new_value = func(epoch_num, old_value)``.
                ``epoch_num`` is the number of epochs that have finished.

        Example:
            Decrease by a factor of 0.9 every two epochs:

            .. code-block:: python

                HyperParamSetterWithFunc('learning_rate',
                                         lambda e, x: x * 0.9 if e % 2 == 0 else x)
        """
        super(HyperParamSetterWithFunc, self).__init__(param)
        self.f = func

    def _get_value_to_set(self):
        return self.f(self.epoch_num, self.get_current_value())


class StatMonitorParamSetter(HyperParamSetter):
    """
    Change the param by monitoring the change of a statistic.
    Change when it wasn't decreasing/increasing enough.
    """
    def __init__(self, param, stat_name, value_func, threshold,
                 last_k, reverse=False):
        """
        Args:
            param: same as in :class:`HyperParamSetter`.
            stat_name (str): name of the statistics.
            value_func (float -> float): a function which returns a new value
                taking the old value.
            threshold (float): change threshold.
            last_k (int): last k epochs.
            reverse (bool): monitor increasing instead of decreasing.

        This callback will change param by ``new_value = value_func(old_value)``, when:
        ``min(stats) >= stats[0] - threshold``, where
        ``stats = [stat_name in last k epochs]``

        Example:
            If validation error wasn't decreasing for 5 epochs, anneal the learning rate:

            .. code-block:: python

                StatMonitorParamSetter('learning_rate', 'val-error', lambda x: x * 0.2, 0, 5)
        """
        super(StatMonitorParamSetter, self).__init__(param)
        self.stat_name = stat_name
        self.value_func = value_func
        self.last_k = last_k
        self.threshold = threshold
        self.reverse = reverse

        self.last_changed_epoch = 0

    def _get_value_to_set(self):
        hist = self.trainer.monitors.get_history(self.stat_name)
        if len(hist) < self.last_k + 1 or \
                self.epoch_num - self.last_changed_epoch < self.last_k:
            return None
        hist = hist[-self.last_k - 1:]    # len==last_k+1

        hist_first = hist[0]
        if not self.reverse:
            hist_min = min(hist)
            if hist_min < hist_first - self.threshold:  # small enough
                return None
        else:
            hist_max = max(hist)
            if hist_max > hist_first + self.threshold:  # large enough
                return None
        self.last_changed_epoch = self.epoch_num
        logger.info("[StatMonitorParamSetter] Triggered, history: " +
                    ','.join(map(str, hist)))
        return self.value_func(self.get_current_value())
