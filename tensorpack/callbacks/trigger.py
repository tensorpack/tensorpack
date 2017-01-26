#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: trigger.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

from abc import abstractmethod, ABCMeta
import six

from .base import Callback, ProxyCallback


__all__ = ['Triggerable', 'PeriodicTrigger']


@six.add_metaclass(ABCMeta)
class Triggerable(Callback):
    """
    Base class for "triggerable" callback. It has a method :meth:`Triggerable.trigger()`
    which can be triggered either inside an epoch or between epochs.
    The higher-level wrapper will take the responsibility to determine when
    to trigger.

    If an triggerable is used as a callback directly (instead of under other
    higher-level wrapper to control the trigger), it will by default trigger after
    every epoch. This is mainly for backward-compatibilty and convenience.
    """

    def trigger(self):
        """
        Trigger something.
        Note that this method may be called both inside an epoch and after an epoch.

        Some operations (e.g. writing scalar stats) currently will cause
        problems if run inside an epoch. This will be fixed in the future.
        """
        # TODO
        self._trigger()

    @abstractmethod
    def _trigger(self):
        pass

    def _trigger_epoch(self):
        """ If used as a callback directly, run the trigger every epoch."""
        self.trigger()


class PeriodicTrigger(ProxyCallback):
    """
    Trigger a :class:`Triggerable` callback every k steps or every k epochs.
    """
    def __init__(self, triggerable, every_k_steps=None, every_k_epochs=None):
        """
        Args:
            triggerable (Triggerable): a Triggerable instance.
            every_k_steps (int): trigger when ``local_step % k == 0``. Set to
                None to disable.
            every_k_epochs (int): trigger when ``epoch_num % k == 0``. Set to
                None to disable.

        every_k_steps and every_k_epochs can be both set, but cannot be both NOne.
        """
        assert isinstance(triggerable, Triggerable), type(triggerable)
        super(PeriodicTrigger, self).__init__(triggerable)
        assert (every_k_epochs is not None) or (every_k_steps is not None), \
            "every_k_steps and every_k_epochs cannot be both None!"
        self._step_k = every_k_steps
        self._epoch_k = every_k_epochs

    def _trigger_step(self, *args):
        if self._step_k is None:
            return
        if self.local_step % self._step_k == 0:
            self.cb.trigger()

    def _trigger_epoch(self, *args):
        if self._epoch_k is None:
            return
        if self.local_step % self._epoch_k == 0:
            self.cb.trigger()
