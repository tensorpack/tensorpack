#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: optimizer.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import tensorflow as tf
from contextlib import contextmanager
from .gradproc import apply_grad_processors as apply_gradproc

__all__ = ['apply_grad_processors', 'ProxyOptimizer',
           'PostProcessOptimizer', 'VariableAssignmentOptimizer']


class ProxyOptimizer(tf.train.Optimizer):
    """
    A transparent proxy which delegates all methods of :class:`tf.train.Optimizer`
    """
    def __init__(self, opt):
        self._opt = opt

    def compute_gradients(self, *args, **kwargs):
        return self._opt.compute_gradients(*args, **kwargs)

    def get_slot(self, *args, **kwargs):
        return self._opt.get_slot(*args, **kwargs)

    def get_slot_names(self, *args, **kwargs):
        return self._opt.get_slot_names(*args, **kwargs)

    def apply_gradients(self, *args, **kwargs):
        return self._opt.apply_gradients(*args, **kwargs)


def apply_grad_processors(opt, gradprocs):
    """
    Wrapper around optimizers to apply gradient processors.

    Args:
        opt (tf.train.Optimizer):
        gradprocs (list[GradientProcessor]): gradient processors to add to the
            optimizer.
    Returns:
        a :class:`tf.train.Optimizer` instance which runs the gradient
        processors before updating the variables.
    """

    class _ApplyGradientProcessor(ProxyOptimizer):
        def __init__(self, opt, gradprocs):
            self._gradprocs = gradprocs
            super(_ApplyGradientProcessor, self).__init__(opt)

        def apply_gradients(self, grads_and_vars,
                            global_step=None, name=None):
            g = apply_gradproc(grads_and_vars, self._gradprocs)
            return self._opt.apply_gradients(g, global_step, name)
    return _ApplyGradientProcessor(opt, gradprocs)


class PostProcessOptimizer(ProxyOptimizer):
    """
    An optimizer which applies some "post-processing operation" per variable
    (e.g. clipping, quantization) after the gradient update.
    """
    def __init__(self, opt, func, colocate=True):
        """
        Args:
            opt (tf.train.Optimizer):
            func (tf.Variable -> tf.Operation or None): the operation needed
                to perform for this variable after the gradient update.
            colocate (boolean): colocate the function with the variable.
        """
        super(PostProcessOptimizer, self).__init__(opt)
        self._func = func
        self._colocate = colocate

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        update_op = super(PostProcessOptimizer, self).apply_gradients(
            grads_and_vars, global_step)
        ops = []
        with tf.control_dependencies([update_op]):
            for _, var in grads_and_vars:
                with self._maybe_colocate(var):
                    op = self._func(var)
                    assert isinstance(op, tf.Operation), op
                    if op is not None:
                        ops.append(op)
        update_op = tf.group(update_op, *ops, name=name)
        return update_op

    @contextmanager
    def _maybe_colocate(self, var):
        G = tf.get_default_graph()
        if self._colocate:
            with G.colocate_with(var):
                yield
        else:
            yield


class VariableAssignmentOptimizer(PostProcessOptimizer):
    """
    An optimizer which assigns each variable a new value (e.g. clipping,
    quantization) after the gradient update.
    """
    def __init__(self, opt, func):
        """
        Args:
            opt (tf.train.Optimizer):
            func (tf.Variable -> tf.Tensor or None): the new value to be
                assigned to this variable after the gradient update.
        """
        def f(v):
            t = func(v)
            if t is None:
                return t
            return tf.assign(v, t, use_locking=False).op
        super(VariableAssignmentOptimizer, self).__init__(opt, f)
