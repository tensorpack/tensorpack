#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: input_source_base.py

from abc import ABCMeta, abstractmethod
import six

from ._utils import get_sublist_by_names, get_tensors_inputs

__all__ = ['InputSource', 'remap_input_source']


@six.add_metaclass(ABCMeta)
class InputSource(object):
    """ Base class for the abstract InputSource. """

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

    def setup(self, inputs_desc):
        """
        Args:
            inputs_desc (list[InputDesc]): list of input desc
        """
        self._setup(inputs_desc)

    def _setup(self, inputs_desc):
        pass

    def get_callbacks(self):
        """
        Returns:
            list[Callback]: extra callbacks required by this InputSource.
        """
        return self._get_callbacks()

    def _get_callbacks(self):
        return []

    def reset_state(self):
        """
        Semantics of this method has not been well defined.
        """
        # TODO
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

    Examples:

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
