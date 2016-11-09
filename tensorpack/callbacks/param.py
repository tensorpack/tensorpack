#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: param.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import tensorflow as tf
from abc import abstractmethod, ABCMeta, abstractproperty
import operator
import six
import os

from .base import Callback
from ..utils import logger
from ..tfutils import get_op_var_name

__all__ = ['HyperParamSetter', 'HumanHyperParamSetter',
           'ScheduledHyperParamSetter',
           'StatMonitorParamSetter',
           'HyperParam', 'GraphVarParam', 'ObjAttrParam']

class HyperParam(object):
    """ Base class for a hyper param"""
    __metaclass__ = ABCMeta

    def setup_graph(self):
        """ setup the graph in `setup_graph` callback stage, if necessary"""
        pass

    @abstractmethod
    def set_value(self, v):
        """ define how the value of the param will be set"""
        pass

    @property
    def readable_name(self):
        """ A name to display"""
        return self._readable_name

class GraphVarParam(HyperParam):
    """ a variable in the graph can be a hyperparam"""
    def __init__(self, name, shape=[]):
        self.name = name
        self.shape = shape
        self._readable_name, self.var_name = get_op_var_name(name)

    def setup_graph(self):
        all_vars = tf.all_variables()
        for v in all_vars:
            if v.name == self.var_name:
                self.var = v
                break
        else:
            raise ValueError("{} is not a VARIABLE in the graph!".format(self.var_name))

        self.val_holder = tf.placeholder(tf.float32, shape=self.shape,
                                         name=self._readable_name + '_feed')
        self.assign_op = self.var.assign(self.val_holder)

    def set_value(self, v):
        self.assign_op.eval(feed_dict={self.val_holder:v})

    def get_value(self):
        return self.var.eval()

class ObjAttrParam(HyperParam):
    """ an attribute of an object can be a hyperparam"""
    def __init__(self, obj, attrname, readable_name=None):
        """ :param readable_name: default to be attrname."""
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

class HyperParamSetter(Callback):
    """
    Base class to set hyperparameters after every epoch.
    """
    __metaclass__ = ABCMeta

    def __init__(self, param):
        """
        :param param: a `HyperParam` instance, or a string (assumed to be a scalar `GraphVarParam`)
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
        :returns: the value to assign to the variable now.
        """
        ret = self._get_value_to_set()
        if ret is not None and ret != self.last_value:
            logger.info("{} at epoch {} will change to {:.8f}".format(
                self.param.readable_name, self.epoch_num + 1, ret))
        self.last_value = ret
        return ret

    def get_current_value(self):
        return self.param.get_value()

    @abstractmethod
    def _get_value_to_set(self):
        pass

    def _trigger_epoch(self):
        self._set_param()

    def _before_train(self):
        self._set_param()

    def _set_param(self):
        v = self.get_value_to_set()
        if v is not None:
            self.param.set_value(v)

class HumanHyperParamSetter(HyperParamSetter):
    """
    Set hyperparameters by loading the value from a file each time it get called.
    """
    def __init__(self, param, file_name='hyper.txt'):
        """
        :param file_name: a file containing the value of the variable.
            Each line in the file is a k:v pair, where k is
            param.readable_name, and v is the value
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
            dic = {str(k):float(v) for k, v in lines}
            ret = dic[self.param.readable_name]
            return ret
        except:
            logger.warn(
                "Cannot find {} in {}".format(
                    self.param.readable_name, self.file_name))
            return None

class ScheduledHyperParamSetter(HyperParamSetter):
    """
    Set hyperparameters by a predefined schedule.
    """
    def __init__(self, param, schedule, interp=None):
        """
        :param schedule: [(epoch1, val1), (epoch2, val2), (epoch3, val3), ...]
            The value is fixed to val1 in epoch [epoch1, epoch2), and so on.
        :param interp: None: no interpolation. 'linear': linear interpolation
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

class StatMonitorParamSetter(HyperParamSetter):
    """
    Set hyperparameter by a func, when a specific stat wasn't
    decreasing/increasing enough in the last $k$ epochs
    """
    def __init__(self, param, stat_name, value_func, threshold,
            last_k, reverse=False
            ):
        """
        Change param by `new_value = value_func(old_value)`,
        if :
            min(stats) >= stats[0] - threshold, where
            stats = [`stat_nam` in latest `last_k` epochs]

        For example, if error wasn't decreasing, anneal the learning rate:
            StatMonitorParamSetter('learning_rate', 'val-error', lambda x: x * 0.2)

        If reverse==True, use 'increasing' instead of decreasing
        """
        super(StatMonitorParamSetter, self).__init__(param)
        self.stat_name = stat_name
        self.value_func = value_func
        self.last_k = last_k
        self.threshold = threshold
        self.reverse = reverse

        self.last_changed_epoch = 0

    def _get_value_to_set(self):
        holder = self.trainer.stat_holder
        hist = holder.get_stat_history(self.stat_name)
        if len(hist) < self.last_k+1 or \
                self.epoch_num - self.last_changed_epoch < self.last_k:
            return None
        hist = hist[-self.last_k-1:]    # len==last_k+1

        hist_first = hist[0]
        if not self.reverse:
            hist_min = min(hist)
            if hist_min < hist_first - self.threshold: # small enough
                return None
        else:
            hist_max = max(hist)
            if hist_max > hist_first + self.threshold: # large enough
                return None
        self.last_changed_epoch = self.epoch_num
        return self.value_func(self.get_current_value())

