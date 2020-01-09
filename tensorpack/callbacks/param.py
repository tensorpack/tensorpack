# -*- coding: utf-8 -*-
# File: param.py


import operator
import os
import numpy as np
from abc import ABCMeta, abstractmethod
from collections import deque
import six

from ..compat import tfv1
from ..tfutils.common import get_op_tensor_name
from ..utils import logger
from .base import Callback

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
    """ A variable in the graph (e.g. learning_rate) can be a hyperparam."""

    def __init__(self, name, shape=()):
        """
        Args:
            name(str): name of the variable.
            shape(tuple): shape of the variable.
        """
        self.name = name
        self.shape = shape
        self._readable_name, self.var_name = get_op_tensor_name(name)

    def setup_graph(self):
        """ Will setup the assign operator for that variable. """
        all_vars = tfv1.global_variables() + tfv1.local_variables()
        for v in all_vars:
            if v.name == self.var_name:
                self.var = v
                break
        else:
            raise ValueError("{} is not a variable in the graph!".format(self.var_name))

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
            readable_name(str): The name to display and set with. Defaults to be ``attrname``.
        """
        self.obj = obj
        self.attrname = attrname
        if readable_name is None:
            self._readable_name = attrname
        else:
            self._readable_name = readable_name

    def set_value(self, v):
        setattr(self.obj, self.attrname, v)

    def get_value(self):
        return getattr(self.obj, self.attrname)


class HyperParamSetter(Callback):
    """
    An abstract base callback to set hyperparameters.

    Once the :meth:`trigger()` method is called,
    the method :meth:`_get_value_to_set` will be used to get a new value for the hyperparameter.
    """

    _chief_only = False

    """
    Also enable this hyperparam setter in the :meth:`before_train` method.
    """
    _enable_before_train = True

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
        self._last_value = None
        self._last_epoch_set = -1

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
        if ret is not None and ret != self._last_value:
            if self.epoch_num != self._last_epoch_set:  # Print this message at most once every epoch
                if self._last_value is None:
                    logger.info("[HyperParamSetter] At global_step={}, {} is set to {:.6f}".format(
                        self.global_step, self.param.readable_name, ret))
                else:
                    logger.info("[HyperParamSetter] At global_step={}, {} changes from {:.6f} to {:.6f}".format(
                        self.global_step, self.param.readable_name, self._last_value, ret))
            self._last_epoch_set = self.epoch_num
            self._last_value = ret
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
        if self._enable_before_train:
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
            file_name(str): a file containing the new value of the parameter.
                Each line in the file is a ``k:v`` pair, for example, ``learning_rate:1e-4``.
                If the pair is not found, the param will not be changed.
        """
        super(HumanHyperParamSetter, self).__init__(param)
        self.file_name = os.path.join(logger.get_logger_dir(), file_name)
        logger.info("Use {} to set hyperparam: '{}'.".format(
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
        except Exception:
            logger.warn(
                "Cannot find {} in {}".format(
                    self.param.readable_name, self.file_name))
            return None


class ScheduledHyperParamSetter(HyperParamSetter):
    """
    Set hyperparameters by a predefined epoch-based schedule.
    """

    def __init__(self, param, schedule, interp=None, step_based=False):
        """
        Args:
            param: same as in :class:`HyperParamSetter`.
            schedule (list): with the format ``[(epoch1, val1), (epoch2, val2), (epoch3, val3)]``.
                Each ``(ep, val)`` pair means to set the param
                to "val" **after** the completion of epoch `ep`.
                If ep == 0, the value will be set before the first epoch
                (because by default the first is epoch 1).
                The epoch numbers have to be increasing.
            interp (str or None): Either None or 'linear'.
                If None, the parameter will only be set when the specific epoch or steps
                is reached exactly. If 'linear', perform linear interpolation (but no extrapolation)
                every time this callback is triggered.
            step_based (bool): interpret ``schedule`` as (step, value) instead
                of (epoch, value).

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
        self._step = step_based
        super(ScheduledHyperParamSetter, self).__init__(param)

    def _get_value_to_set(self):  # override parent
        return self._get_value_to_set_at_point(self._current_point())

    def _current_point(self):
        return self.global_step if self._step else self.epoch_num

    def _check_value_at_beginning(self):
        v = None
        # we are at `before_train`, therefore the epoch/step associated with `current_point` has finished.
        for p in range(0, self._current_point() + 1):
            v = self._get_value_to_set_at_point(p) or v
        actual_value = self.param.get_value()
        if v is not None and not np.isclose(v, actual_value):
            logger.warn("According to scheduler {}, parameter '{}' should become {} at the current point. "
                        "However its current value is {}. "
                        "If this is the only scheduler being used, you may want to check whether your "
                        "initialization of the parameter is as expected".format(
                            self, self.param.readable_name, v, actual_value))

    def _get_value_to_set_at_point(self, point):
        """
        Using schedule, compute the value to be set at a given point.
        """
        laste, lastv = None, None
        for e, v in self.schedule:
            if e == point:
                return v    # meet the exact boundary, return directly
            if e > point:
                break
            laste, lastv = e, v
        if laste is None or laste == e:
            # hasn't reached the first scheduled point, or reached the end of all scheduled points
            return None
        if self.interp is None:
            # If no interpolation, nothing to do.
            return None
        v = (point - laste) * 1. / (e - laste) * (v - lastv) + lastv
        return v

    def _before_train(self):
        super(ScheduledHyperParamSetter, self)._before_train()
        self._check_value_at_beginning()

    def _trigger_epoch(self):
        if not self._step:
            self.trigger()

    def _trigger_step(self):
        if self._step:
            self.trigger()

    def __str__(self):
        return "ScheduledHyperParamSetter(schedule={})".format(self.schedule)


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
    Change the param by monitoring the change of a scalar statistics.
    The param will be changed when the scalar does not decrease/increase enough.

    Once triggered, this callback observes the latest **one** value of ``stat_name``, from the monitor backend.

    This callback will then change a hyperparameter ``param`` by ``new_value = value_func(old_value)``, if:
    ``min(history) >= history[0] - threshold``, where
    ``history = [the most recent k observations of stat_name]``

    Note:
        The statistics of interest must be created at a frequency higher than or equal to this callback.
        For example, using ``PeriodicTrigger(StatMonitorParamSetter(...), every_k_steps=100)``
        is meaningless if the statistics to be monitored is only updated every 500 steps.

        Callbacks are executed in order. Therefore, if the statistics to be monitored
        is created after this callback, the behavior of this callback may get delayed.

    Example:

        If validation error wasn't decreasing for 5 epochs, decay the learning rate by 0.2:

        .. code-block:: python

            StatMonitorParamSetter('learning_rate', 'val-error',
                                    lambda x: x * 0.2, threshold=0, last_k=5)
    """

    _enable_before_train = False

    def __init__(self, param, stat_name, value_func, threshold,
                 last_k, reverse=False):
        """
        Args:
            param: same as in :class:`HyperParamSetter`.
            stat_name (str): name of the statistics.
            value_func (float -> float): a function which returns a new value
                taking the old value.
            threshold (float): change threshold.
            last_k (int): use last k observations of statistics.
            reverse (bool): monitor increasing instead of decreasing.
                If True, ``param`` will be changed when ``max(history) <= history[0] + threshold``.
        """
        super(StatMonitorParamSetter, self).__init__(param)
        self.stat_name = stat_name
        self.value_func = value_func
        self.history = deque(maxlen=last_k)
        self.threshold = threshold
        self.reverse = reverse

    def _get_value_to_set(self):
        try:
            last = self.trainer.monitors.get_history(self.stat_name)[-1]
        except (KeyError, IndexError):
            logger.warn(
                "[StatMonitorParamSetter] No history data available for key '{}'.".format(self.stat_name))
            return None
        if len(self.history) and last[0] == self.history[-1][0]:
            logger.warn("StatMonitorParamSetter is triggered, but no new data has been added since last time.")
            return None

        self.history.append(last)

        if len(self.history) < self.history.maxlen:
            return None

        values = [k[1] for k in self.history]
        hist_first = values[0]
        if not self.reverse:
            hist_min = min(values)
            if hist_min < hist_first - self.threshold:  # small enough
                return None
        else:
            hist_max = max(values)
            if hist_max > hist_first + self.threshold:  # large enough
                return None
        self.history.clear()
        logger.info(
            "[StatMonitorParamSetter] Triggered, history of {}: ".format(
                self.stat_name) + ','.join([str(round(x, 3)) for x in values]))
        return self.value_func(self.get_current_value())
