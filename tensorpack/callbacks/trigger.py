#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: trigger.py

from .base import ProxyCallback, Callback

__all__ = ['PeriodicTrigger', 'PeriodicCallback', 'EnableCallbackIf']


class PeriodicTrigger(ProxyCallback):
    """
    Schedule to trigger a callback every k global steps or every k epochs by its :meth:`trigger()` method.
    Most existing callbacks which do something every epoch are implemented
    with :meth:`trigger()` method.

    All other methods (``before/after_run``, ``trigger_step``, etc) are unaffected.
    """

    _chief_only = False

    def __init__(self, triggerable, every_k_steps=None, every_k_epochs=None):
        """
        Args:
            triggerable (Callback): a Callback instance with a trigger method to be called.
            every_k_steps (int): trigger when ``global_step % k == 0``. Set to
                None to ignore.
            every_k_epochs (int): trigger when ``epoch_num % k == 0``. Set to
                None to ignore.

        every_k_steps and every_k_epochs can be both set, but cannot be both None.
        """
        assert isinstance(triggerable, Callback), type(triggerable)
        super(PeriodicTrigger, self).__init__(triggerable)
        assert (every_k_epochs is not None) or (every_k_steps is not None), \
            "every_k_steps and every_k_epochs cannot be both None!"
        self._step_k = every_k_steps
        self._epoch_k = every_k_epochs

    def _trigger_step(self):
        self.cb.trigger_step()
        if self._step_k is None:
            return
        if self.global_step % self._step_k == 0:
            self.cb.trigger()

    def _trigger_epoch(self):
        if self._epoch_k is None:
            return
        if self.epoch_num % self._epoch_k == 0:
            self.cb.trigger()

    def __str__(self):
        return "PeriodicTrigger-" + str(self.cb)


class PeriodicRunHooks(ProxyCallback):
    """
    Enable the ``{before,after}_run`` methods of a callback every k global steps.
    All other methods are untouched.
    """

    _chief_only = False

    def __init__(self, callback, every_k_steps):
        """
        Args:
            callback (Callback):
            every_k_steps(int): call ``{before,after}_run`` when
                ``global_step % k == 0``.
        """
        self._every_k_steps = int(every_k_steps)
        super(PeriodicRunHooks, self).__init__(callback)

    def _before_run(self, ctx):
        if self.global_step % self._every_k_steps == 0:
            self._enabled = True
            return self.cb._before_run(ctx)
        else:
            self._enabled = False

    def _after_run(self, ctx, rv):
        if self._enabled:
            self.cb._after_run(ctx, rv)

    def __str__(self):
        return "PeriodicRunHooks-" + str(self.cb)


class EnableCallbackIf(ProxyCallback):
    """
    Enable ``{before,after}_epoch``, ``{before,after}_run``,
    ``trigger_{epoch,step}``
    methods of a callback, only when some condition satisfies.
    The other methods are unaffected.

    Note:
        If you use ``{before,after}_run``,
        ``pred`` will be evaluated only in ``before_run``.
    """

    _chief_only = False

    def __init__(self, callback, pred):
        """
        Args:
            callback (Callback):
            pred (self -> bool): a callable predicate. Has to be a pure function.
        """
        self._pred = pred
        super(EnableCallbackIf, self).__init__(callback)

    def _before_run(self, ctx):
        if self._pred(self):
            self._enabled = True
            return super(EnableCallbackIf, self)._before_run(ctx)
        else:
            self._enabled = False

    def _after_run(self, ctx, rv):
        if self._enabled:
            super(EnableCallbackIf, self)._after_run(ctx, rv)

    def _before_epoch(self):
        if self._pred(self):
            super(EnableCallbackIf, self)._before_epoch()

    def _after_epoch(self):
        if self._pred(self):
            super(EnableCallbackIf, self)._after_epoch()

    def _trigger_epoch(self):
        if self._pred(self):
            super(EnableCallbackIf, self)._trigger_epoch()

    def _trigger_step(self):
        if self._pred(self):
            super(EnableCallbackIf, self)._trigger_step()

    def __str__(self):
        return "EnableCallbackIf-" + str(self.cb)


class PeriodicCallback(EnableCallbackIf):
    """
    Make the calls to the following methods of a callback **less** frequent:
    ``{before,after}_epoch``, ``{before,after}_run``, ``trigger_{epoch,step}``.

    These methods will be enabled only when ``global_step % every_k_steps == 0`
    or ``epoch_num % every_k_epochs == 0``. The other methods are unaffected.

    Note that this can only makes a callback **less** frequent than before.
    :class:`PeriodicTrigger` can
    make a callback which supports :meth:`trigger()` method more frequent than before.
    """

    _chief_only = False

    def __init__(self, callback, every_k_steps=None, every_k_epochs=None):
        """
        Args:
            callback (Callback): a Callback instance.
            every_k_steps (int): enable the callback when ``global_step % k == 0``. Set to
                None to ignore.
            every_k_epochs (int): enable the callback when ``epoch_num % k == 0``.
                Also enable when the last step finishes (``epoch_num == max_epoch``
                and ``local_step == steps_per_epoch - 1``). Set to None to ignore.

        every_k_steps and every_k_epochs can be both set, but cannot be both None.
        """
        assert isinstance(callback, Callback), type(callback)
        assert (every_k_epochs is not None) or (every_k_steps is not None), \
            "every_k_steps and every_k_epochs cannot be both None!"
        self._step_k = every_k_steps
        self._epoch_k = every_k_epochs
        super(PeriodicCallback, self).__init__(callback,
                                               PeriodicCallback.predicate)

    def predicate(self):
        if self._step_k is not None and self.global_step % self._step_k == 0:
            return True
        if self._epoch_k is not None and self.epoch_num % self._epoch_k == 0:
            return True
        if self._epoch_k is not None:
            if self.local_step == self.trainer.steps_per_epoch - 1 and \
                    self.epoch_num == self.trainer.max_epoch:
                return True
        return False

    def __str__(self):
        return "PeriodicCallback-" + str(self.cb)
