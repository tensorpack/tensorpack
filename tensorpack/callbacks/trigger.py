#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: trigger.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

from .base import ProxyCallback, Triggerable


__all__ = ['PeriodicTrigger', 'PeriodicCallback']


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

    def _trigger_step(self):
        if self._step_k is None:
            return
        # trigger_step is triggered after run_step, so
        # local_step + 1 is the number of step that have finished
        if (self.trainer.local_step + 1) % self._step_k == 0:
            self.cb.trigger()

    def _trigger_epoch(self):
        if self._epoch_k is None:
            return
        if self.epoch_num % self._epoch_k == 0:
            self.cb.trigger()

    def __str__(self):
        return "PeriodicTrigger-" + str(self.cb)


class PeriodicCallback(ProxyCallback):
    """
    Wrap a callback so that after every ``period`` epochs, its :meth:`trigger_epoch`
    method is called.

    Note that this wrapper will proxy the :meth:`trigger_step` method as-is.
    To schedule a :class:`Triggerable` callback more frequent than once per
    epoch, use :class:`PeriodicTrigger` instead.
    """

    def __init__(self, cb, period):
        """
        Args:
            cb(Callback): the callback to be triggered periodically
            period(int): the period, the number of epochs for a callback to be triggered.
        """
        super(PeriodicCallback, self).__init__(cb)
        self.period = int(period)

    def _trigger_epoch(self):
        if self.epoch_num % self.period == 0:
            self.cb.trigger_epoch()

    def __str__(self):
        return "Periodic-" + str(self.cb)
