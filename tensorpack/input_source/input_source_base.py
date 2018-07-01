# -*- coding: utf-8 -*-
# File: input_source_base.py

from abc import ABCMeta, abstractmethod
import copy
import six
from six.moves import zip
from contextlib import contextmanager
import tensorflow as tf

from ..utils.argtools import memoized, call_only_once
from ..callbacks.base import CallbackFactory
from ..tfutils.common import get_op_tensor_name
from ..utils import logger

__all__ = ['InputSource', 'remap_input_source']


def get_tensors_inputs(placeholders, tensors, names):
    """
    Args:
        placeholders (list[Tensor]):
        tensors (list[Tensor]): list of tf.Tensor
        names (list[str]): names matching the tensors

    Returns:
        list[Tensor]: inputs to used with build_graph(),
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
            list: A list of tensors corresponding to the inputs of the model,
                used as input of :func:`build_graph`.
                For non-placeholder tensors, should always create and return new tensors when called.
        """
        return self._get_input_tensors()

    @abstractmethod
    def _get_input_tensors(self):
        pass

    @call_only_once
    def setup(self, inputs_desc):
        """
        Args:
            inputs_desc (list[InputDesc]): list of input desc

        Returns:
            list[Callback]: extra callbacks needed by this InputSource.
                callbacks of InputSource cannot use any `trigger*()` method.
        """
        self._setup(inputs_desc)
        self._setup_done = True
        return self.get_callbacks()

    def _setup(self, inputs_desc):
        pass

    def setup_done(self):
        """
        Returns:
            bool: whether :meth:`setup()` has been called.
        """
        return self._setup_done

    @memoized
    def get_callbacks(self):
        """
        An InputSource might need some extra maintenance during training,
        which is done also through the Callback interface.
        This method returns the callbacks and the return value will be memoized.

        All callbacks will be automatically marked as `chief_only=False`,
        so they will run on all nodes.

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

    def _setup(self, inputs_desc):
        self._input.setup(inputs_desc)

    def _get_callbacks(self):
        return self._input.get_callbacks()

    def _size(self):
        return self._input.size()

    def _reset_state(self):
        self._input.reset_state()


def remap_input_source(input, names):
    """
    When you have some :class:`InputSource` which doesn't match the inputs in
    your :class:`ModelDesc`, use `RemapInputSource`.
    It produces placeholders for all the inputs in your model,
    except that the corresponding ones are replaced with the tensor produced
    by the given :class:`InputSource`.

    Args:
        input(InputSource): a :class:`InputSource`, whose tensors will get mapped.
        names(list[str]): list of input names corresponding to the tensors
            produced by ``input``.

    Returns:
        InputSource:

    Example:

    .. code-block:: python

        input1 = QueueInput(ds)
        # assume ds produces 'image' and 'label', but the graph takes more
        # inputs for some reasons, or takes inputs of a different order:
        inputs_desc = [InputDesc(tf.float32, (None,10), 'score'),
                       InputDesc(tf.float32, (None,20,20,3), 'label'),
                       InputDesc(tf.int32, (None,), 'image') ]
        input2 = remap_input_source(input1, ['image', 'label'])
        input2.setup(inputs_desc)
        # now, input2.get_input_tensors() will return a placeholder for 'score',
        # plus the tensors returned by input1.get_input_tensors()
    """
    def __init__(self, input, names):
        ProxyInputSource.__init__(self, input)
        assert isinstance(names, (list, tuple)), names
        self._names = tuple(names)

    def _setup(self, inputs):
        self._all_placehdrs = [v.build_placeholder_reuse() for v in inputs]
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
