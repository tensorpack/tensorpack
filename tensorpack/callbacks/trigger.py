#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: trigger.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

from abc import abstractmethod, ABCMeta
import six

from .base import Callback


__all__ = ['Triggerable']


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
