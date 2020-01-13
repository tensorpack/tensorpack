# -*- coding: utf-8 -*-
# File: input_source_base.py

import copy
from abc import ABCMeta, abstractmethod
from contextlib import contextmanager
import six
import tensorflow as tf

from ..callbacks.base import CallbackFactory
from ..tfutils.common import get_op_tensor_name
from ..utils import logger
from ..utils.argtools import call_only_once, memoized_method
from ..compat import tfv1

__all__ = ['InputSource', 'remap_input_source']


def build_or_reuse_placeholder(tensor_spec):
    """
    Build a tf.placeholder from the metadata in the given tensor spec, or return an existing one.

    Args:
        tensor_spec (tf.TensorSpec):

    Returns:
        tf.Tensor:
    """
    g = tfv1.get_default_graph()
    name = tensor_spec.name
    try:
        tensor = g.get_tensor_by_name(name + ':0')
        assert "Placeholder" in tensor.op.type, "Tensor {} exists but is not a placeholder!".format(name)
        assert tensor_spec.is_compatible_with(tensor), \
            "Tensor {} exists but is not compatible with the signature!".format(tensor)
        if tensor.shape.as_list() == tensor_spec.shape.as_list():
            # It might be desirable to use a placeholder of a different shape in some tower
            # (e.g., a less specific shape)

            # Comparing `tensor.shape` directly doesn't work, because
            # tensorflow thinks `tf.Dimension(None)` and `tf.Dimension(None)` are not equal.
            return tensor
    except KeyError:
        pass
    with tfv1.name_scope(None):   # clear any name scope it might get called in
        ret = tfv1.placeholder(
            tensor_spec.dtype, shape=tensor_spec.shape, name=tensor_spec.name)
    return ret


def get_tensors_inputs(placeholders, tensors, names):
    """
    Args:
        placeholders (list[Tensor]):
        tensors (list[Tensor]): list of tf.Tensor
        names (list[str]): names matching the given tensors

    Returns:
        list[Tensor]: inputs to used for the tower function,
            with the corresponding placeholders replaced by tensors.
    """
    assert len(tensors) == len(names), \
        "Input tensors {} and input names {} have different length!".format(
            tensors, names)
    ret = copy.copy(placeholders)
    placeholder_names = [p.name for p in placeholders]
    for name, tensor in zip(names, tensors):
        tensorname = get_op_tensor_name(name)[1]
        try:
            idx = placeholder_names.index(tensorname)
        except ValueError:
            logger.error("Name {} is not a model input!".format(tensorname))
            raise
        ret[idx] = tensor
    return ret


def get_sublist_by_names(lst, names):
    """
    Args:
        lst (list): list of objects with "name" property.

    Returns:
        list: a sublist of objects, matching names
    """
    orig_names = [p.name for p in lst]
    ret = []
    for name in names:
        try:
            idx = orig_names.index(name)
        except ValueError:
            logger.error("Name {} doesn't appear in lst {}!".format(
                name, str(orig_names)))
            raise
        ret.append(lst[idx])
    return ret


@six.add_metaclass(ABCMeta)
class InputSource(object):
    """ Base class for the abstract InputSource. """

    _name_scope = None
    _setup_done = False

    def get_input_tensors(self):
        """
        Returns:
            list[Tensor]: A list of tensors corresponding to the inputs of the model.
                Will be used as input for the tower function.
                This method should always create and return new tensors when called,
                unless it returns placeholders.
        """
        return self._get_input_tensors()

    @abstractmethod
    def _get_input_tensors(self):
        pass

    @call_only_once
    def setup(self, input_signature):
        """
        Args:
            input_signature (list[tf.TensorSpec]): list of specs for each input tensor

        Returns:
            list[Callback]: extra callbacks needed by this InputSource.
            callbacks of InputSource cannot use any `trigger*()` method.
        """
        self._setup(input_signature)
        self._setup_done = True
        return self.get_callbacks()

    def _setup(self, input_signature):
        pass

    def setup_done(self):
        """
        Returns:
            bool: whether :meth:`setup()` has been called.
        """
        return self._setup_done

    @memoized_method
    def get_callbacks(self):
        """
        An InputSource might need some extra maintenance during training,
        which is done also through the Callback interface.
        This method returns the callbacks and the return value will be memoized.

        All callbacks will be automatically marked as `chief_only=False`,
        so they will run on all nodes.

        Callbacks returned by :class:`InputSource` only supports a subset of callback's functionalities:

        1. It cannot access the trainer, because an :class:`InputSource` can be used in pure inference.
        2. It cannot use the following methods: `trigger_{step,epoch}, {before,after}_epoch`.

        In other words, these callbacks should only have the basic functionality of `tf.train.SessionRunHooks`.

        Returns:
            list[Callback]: extra callbacks needed by this InputSource.
        """
        assert self.setup_done()
        ret = [CallbackFactory(
            before_train=lambda _: self.reset_state())] + self._get_callbacks()

        for r in ret:
            r.set_chief_only(False)    # no input callbacks should be chief-only
        return ret

    def _get_callbacks(self):
        return []

    def reset_state(self):
        """
        Initialize/reinitialize this InputSource.
        Must be called under a default session.

        For training, it will get called once by the trainer in `before_train` callbacks.
        For inference, the :class:`InferenceRunner` will call this method each time it is triggered.
        """
        self._reset_state()

    def _reset_state(self):
        pass

    def size(self):
        """
        Returns:
            int: epoch size of the InputSource
        """
        return self._size()

    def _size(self):
        raise NotImplementedError()

    @contextmanager
    def cached_name_scope(self):
        """
        Yield a context under a cached name scope, whose name is the name of
        this InputSource class.
        """
        if self._name_scope:
            with tf.name_scope(self._name_scope):
                yield self._name_scope
        else:
            name = type(self).__name__
            with tf.name_scope(name) as ns:
                self._name_scope = ns
                yield ns


class ProxyInputSource(InputSource):
    """
    An InputSource which proxy every method to ``self._input``.
    """
    def __init__(self, input):
        assert isinstance(input, InputSource), input
        self._input = input

    def _get_input_tensors(self):
        return self._input.get_input_tensors()

    def _setup(self, input_signature):
        self._input.setup(input_signature)

    def _get_callbacks(self):
        return self._input.get_callbacks()

    def _size(self):
        return self._input.size()

    def _reset_state(self):
        self._input.reset_state()


def remap_input_source(input, names):
    """
    When you have some :class:`InputSource` which doesn't match the inputs of
    your tower function, use `RemapInputSource`.
    It produces placeholders for all the inputs in your model,
    except that the corresponding ones are replaced with the tensor produced
    by the given :class:`InputSource`.

    Example:

    .. code-block:: python

        input1 = QueueInput(ds)
        # assume ds produces data that should be fed to 'image' and 'label',
        # but the graph takes more inputs for some reasons, or takes inputs
        # of a different order, for example like the following:

        # input_signature = [tf.TensorSpec((None,10), tf.float32, 'score'),
        #                    tf.TensorSpec((None,20,20,3), tf.float32, 'label'),
        #                    tf.TensorSpec((None,), tf.int32, 'image') ]

        input2 = remap_input_source(input1, ['image', 'label'])
        # now, if input2 is used with the above input_signature, it will return a
        # placeholder for 'score', plus the tensors returned by input1
    """
    def __init__(self, input, names):
        """
        Args:
            input(InputSource): a :class:`InputSource`, whose tensors will get mapped.
            names(list[str]): list of input names corresponding to the tensors
                produced by ``input``.

        Returns:
            InputSource:
        """
        ProxyInputSource.__init__(self, input)
        assert isinstance(names, (list, tuple)), names
        self._names = tuple(names)

    def _setup(self, inputs):
        self._all_placehdrs = [build_or_reuse_placeholder(v) for v in inputs]
        inputs_subset = get_sublist_by_names(inputs, self._names)
        self._input.setup(inputs_subset)

    def _get_input_tensors(self):
        ret = self._input.get_input_tensors()
        assert len(ret) == len(self._names)
        return get_tensors_inputs(
            self._all_placehdrs, ret, self._names)

    oldcls = type(input)
    # inherit oldcls so that type check in various places would work
    cls = type('Remapped' + oldcls.__name__, (ProxyInputSource, oldcls), {
        '__init__': __init__,
        '_setup': _setup,
        '_get_input_tensors': _get_input_tensors})
    return cls(input, names)
