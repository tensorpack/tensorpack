# -*- coding: UTF-8 -*-
# File: base.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import tensorflow as tf
from abc import ABCMeta
import six

__all__ = ['Callback', 'PeriodicCallback', 'ProxyCallback', 'CallbackFactory']


@six.add_metaclass(ABCMeta)
class Callback(object):
    """ Base class for all callbacks """

    def setup_graph(self, trainer):
        """
        Called before finalizing the graph.
        Use this callback to setup some ops used in the callback.

        Args:
            trainer(Trainer): the trainer which calls the callback
        """
        self.trainer = trainer
        self.graph = tf.get_default_graph()
        self.epoch_num = self.trainer.config.starting_epoch - 1
        # self.epoch_num is always the number of epochs that finished updating parameters.
        with tf.name_scope(type(self).__name__):
            self._setup_graph()

    def _setup_graph(self):
        pass

    def before_train(self):
        """
        Called right before the first iteration.
        """
        self._before_train()

    def _before_train(self):
        pass

    def trigger_step(self):
        """
        Callback to be triggered after every step (every backpropagation)

        Could be useful to apply some tricks on parameters (clipping, low-rank, etc)
        """

    def trigger_epoch(self):
        """
        Triggered after every epoch.

        In this function, ``self.epoch_num`` would be the number of epoch finished.
        """
        self.epoch_num += 1
        self._trigger_epoch()

    def _trigger_epoch(self):
        pass

    def after_train(self):
        """
        Called after training.
        """
        self._after_train()

    def _after_train(self):
        pass

    def __str__(self):
        return type(self).__name__


class ProxyCallback(Callback):
    """ A callback which proxy all methods to another callback.
        It's useful as a base class of callbacks which decorate other callbacks.
    """

    def __init__(self, cb):
        """
        Args:
            cb(Callback): the underlying callback
        """
        self.cb = cb

    def _before_train(self):
        self.cb.before_train()

    def _setup_graph(self):
        self.cb.setup_graph(self.trainer)

    def _after_train(self):
        self.cb.after_train()

    def _trigger_epoch(self):
        self.cb.trigger_epoch()

    def __str__(self):
        return "Proxy-" + str(self.cb)


class PeriodicCallback(ProxyCallback):
    """
    Wrap a callback so that it is triggered after every ``period`` epochs.

    Doesn't work for ``trigger_step``.
    """

    def __init__(self, cb, period):
        """
        Args:
            cb(Callback): the callback to be triggered periodically
            period(int): the period

        Note:
            In ``cb``, ``self.epoch_num`` will not be the true number of
            epochs any more.
        """
        super(PeriodicCallback, self).__init__(cb)
        self.period = int(period)

    def _trigger_epoch(self):
        if self.epoch_num % self.period == 0:
            self.cb.epoch_num = self.epoch_num - 1
            self.cb.trigger_epoch()

    def __str__(self):
        return "Periodic-" + str(self.cb)


class CallbackFactory(Callback):
    """
    Create a callback with some lambdas.
    """
    def __init__(self, setup_graph=None, before_train=None,
                 trigger_epoch=None, after_train=None):
        """
        Each lambda takes ``self`` as the only argument.
        """

        self._cb_setup_graph = setup_graph
        self._cb_before_train = before_train
        self._cb_trigger_epoch = trigger_epoch
        self._cb_after_train = after_train

    def _setup_graph(self):
        if self._cb_setup_graph:
            self._cb_setup_graph(self)

    def _before_train(self):
        if self._cb_before_train:
            self._cb_before_train(self)

    def _trigger_epoch(self):
        if self._cb_trigger_epoch:
            self._cb_trigger_epoch(self)

    def _after_train(self):
        if self._cb_after_train:
            self._cb_after_train(self)
