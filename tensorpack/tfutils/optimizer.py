#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: optimizer.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import tensorflow as tf
from .gradproc import apply_grad_processors as apply_gradproc

__all__ = ['apply_grad_processors', 'ProxyOptimizer',
           'PostProcessVariablesOptimizer']


class ProxyOptimizer(tf.train.Optimizer):
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


class PostProcessVariablesOptimizer(ProxyOptimizer):
    """
    An optimizer which applies an operation to variables (e.g. clipping,
    quantization) after updating the gradient.
    """
    def __init__(self, opt, func, colocate=True):
        """
        Args:
            opt (tf.train.Optimizer):
            func (tf.Variable -> tf.Operation or None): the operation needed
                to perform for this variable after the gradient update.
            colocate (boolean): colocate the function with the variable.
        """
        super(PostProcessVariablesOptimizer, self).__init__(opt)
        self._func = func
        self._colocate = colocate

    def apply_gradients(self, grads_and_vars, global_step=None, name=None):
        update_op = super(PostProcessVariablesOptimizer, self).apply_gradients(
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

    def _maybe_colocate(self, var):
        G = tf.get_default_graph()
        if self._colocate:
            with G.colocate_with(var):
                yield
        else:
            yield
