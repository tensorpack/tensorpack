#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: hooks.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

""" Compatible layers between tf.train.SessionRunHook and Callback"""

import tensorflow as tf
from .base import Callback


__all__ = ['CallbackToHook', 'HookToCallback']


class CallbackToHook(tf.train.SessionRunHook):
    """ This is only for internal implementation of
        before_run/after_run callbacks.
        You shouldn't need to use this.
    """
    def __init__(self, cb):
        self._cb = cb

    def before_run(self, ctx):
        return self._cb.before_run(ctx)

    def after_run(self, ctx, vals):
        self._cb.after_run(ctx, vals)


class HookToCallback(Callback):
    """
    Make a ``tf.train.SessionRunHook`` into a callback.
    Note that the `coord` argument in `after_create_session` will be None.
    """
    def __init__(self, hook):
        """
        Args:
            hook (tf.train.SessionRunHook):
        """
        self._hook = hook

    def _setup_graph(self):
        with tf.name_scope(None):   # jump out of the name scope
            self._hook.begin()

    def _before_train(self):
        sess = tf.get_default_session()
        # TODO fix coord?
        self._hook.after_create_session(sess, None)

    def _before_run(self, ctx):
        return self._hook.before_run(ctx)

    def _after_run(self, ctx, run_values):
        self._hook.after_run(ctx, run_values)

    def _after_train(self):
        self._hook.end()
