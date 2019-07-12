# -*- coding: utf-8 -*-
# File: varreplace.py
# Credit: Qinyao He

from contextlib import contextmanager

from ..compat import tfv1 as tf
from .common import get_tf_version_tuple

__all__ = ['custom_getter_scope', 'freeze_variables', 'remap_variables']


@contextmanager
def custom_getter_scope(custom_getter):
    """
    Args:
        custom_getter: the same as in :func:`tf.get_variable`

    Returns:
        The current variable scope with a custom_getter.
    """
    scope = tf.get_variable_scope()
    if get_tf_version_tuple() >= (1, 5):
        with tf.variable_scope(
                scope, custom_getter=custom_getter,
                auxiliary_name_scope=False):
            yield
    else:
        ns = tf.get_default_graph().get_name_scope()
        with tf.variable_scope(
                scope, custom_getter=custom_getter):
            with tf.name_scope(ns + '/' if ns else ''):
                yield


def remap_variables(fn):
    """
    Use fn to map the output of any variable getter.

    Args:
        fn (tf.Variable -> tf.Tensor)

    Returns:
        The current variable scope with a custom_getter that maps
        all the variables by fn.

    Example:
        .. code-block:: python

            from tensorpack.tfutils import varreplace
            with varreplace.remap_variables(lambda var: quantize(var)):
                x = FullyConnected('fc', x, 1000)   # fc/{W,b} will be quantized
    """
    def custom_getter(getter, *args, **kwargs):
        v = getter(*args, **kwargs)
        return fn(v)
    return custom_getter_scope(custom_getter)


def freeze_variables(stop_gradient=True, skip_collection=False):
    """
    Return a context to freeze variables,
    by wrapping ``tf.get_variable`` with a custom getter.
    It works by either applying ``tf.stop_gradient`` on the variables,
    or keeping them out of the ``TRAINABLE_VARIABLES`` collection, or
    both. Both options have their own pros and cons.

    Example:
        .. code-block:: python

            from tensorpack.tfutils import varreplace
            with varreplace.freeze_variable(stop_gradient=False, skip_collection=True):
                x = FullyConnected('fc', x, 1000)   # fc/* will not be trained

    Args:
        stop_gradient (bool): if True, variables returned from `get_variable`
            will be wrapped with `tf.stop_gradient`.

            Note that the created variables may still have gradient when accessed
            by other approaches (e.g. by name, or by collection).
            For example, they may still have a gradient in weight decay.
            Also note that this makes `tf.get_variable` returns a Tensor instead of a Variable,
            which may break existing contract.
            Therefore, it's recommended to use the `skip_collection` option instead.
        skip_collection (bool): if True, do not add the variable to
            ``TRAINABLE_VARIABLES`` collection, but to ``MODEL_VARIABLES``
            collection. As a result they will not be trained by default.

    Note:

    `stop_gradient` only stops variables returned by `get_variable` **within the context** to
    contribute no gradient in this context. Therefore it may not completely freeze the variables.
    For example:

        1. If a variable is created, or reused outside of the context, it can still contribute to the
           gradient of other tensors.
        2. If a freezed variable is accessed by other approaches (e.g., by names, by collections),
          it can still contribute to the gradient of other tensors.
          For example, weight decay cannot be stopped by a `stop_gradient` context.

    `skip_collection` has to be used the first time the variable is created.
    Once `skip_collection` is used, the variable is not a trainable variable anymore,
    and will be completely freezed from gradient update in tensorpack's single-cost trainer.

    Choose the option carefully depend on what you need.
    """
    def custom_getter(getter, *args, **kwargs):
        trainable = kwargs.get('trainable', True)
        name = args[0] if len(args) else kwargs.get('name')
        if skip_collection:
            kwargs['trainable'] = False
        v = getter(*args, **kwargs)
        if skip_collection:
            tf.add_to_collection(tf.GraphKeys.MODEL_VARIABLES, v)
        if trainable and stop_gradient:
            v = tf.stop_gradient(v, name='freezed_' + name)
        return v
    return custom_getter_scope(custom_getter)
