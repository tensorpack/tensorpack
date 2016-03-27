#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: param.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import tensorflow as tf
from abc import abstractmethod, ABCMeta
import operator

from .base import Callback
from ..utils import logger, get_op_var_name

__all__ = ['HyperParamSetter', 'HumanHyperParamSetter',
           'ScheduledHyperParamSetter']

class HyperParamSetter(Callback):
    __metaclass__ = ABCMeta

    # TODO maybe support InputVar?
    def __init__(self, var_name, shape=[]):
        self.op_name, self.var_name = get_op_var_name(var_name)
        self.shape = shape
        self.last_value = None

    def _before_train(self):
        all_vars = tf.all_variables()
        for v in all_vars:
            if v.name == self.var_name:
                self.var = v
                break
        else:
            raise ValueError("{} is not a VARIABLE in the graph!".format(self.var_name))

        self.val_holder = tf.placeholder(tf.float32, shape=self.shape,
                                         name=self.op_name + '_feed')
        self.assign_op = self.var.assign(self.val_holder)

    def get_current_value(self):
        ret = self._get_current_value()
        if ret is not None and ret != self.last_value:
            logger.info("{} at epoch {} is changed to {}".format(
                self.op_name, self.epoch_num, ret))
        self.last_value = ret
        return ret

    @abstractmethod
    def _get_current_value(self):
        pass

    def _trigger_epoch(self):
        v = self.get_current_value()
        if v is not None:
            self.assign_op.eval(feed_dict={self.val_holder:v})

class HumanHyperParamSetter(HyperParamSetter):
    def __init__(self, var_name, file_name):
        """
        read value from file_name.
        file_name: each line in the file is a k:v pair
        """
        self.file_name = file_name
        super(HumanHyperParamSetter, self).__init__(var_name)

    def  _get_current_value(self):
        with open(self.file_name) as f:
            lines = f.readlines()
        lines = [s.strip().split(':') for s in lines]
        dic = {str(k):float(v) for k, v in lines}
        return dic[self.op_name]

class ScheduledHyperParamSetter(HyperParamSetter):
    def __init__(self, var_name, schedule):
        """
        schedule: [(epoch1, val1), (epoch2, val2), (epoch3, val3), ...]
        """
        self.schedule = sorted(schedule, key=operator.itemgetter(0))
        super(ScheduledHyperParamSetter, self).__init__(var_name)

    def _get_current_value(self):
        for e, v in self.schedule:
            if e == self.epoch_num:
                return v
        return None



