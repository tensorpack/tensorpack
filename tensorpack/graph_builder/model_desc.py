# -*- coding: utf-8 -*-
# File: model_desc.py


from collections import namedtuple
import tensorflow as tf

from ..utils.argtools import memoized_method
from ..utils.develop import deprecated
from ..tfutils.common import get_op_tensor_name
from ..compat import backport_tensor_spec, tfv1

TensorSpec = backport_tensor_spec()


__all__ = ['InputDesc', 'ModelDesc', 'ModelDescBase']


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
        return tensor
    except KeyError:
        with tfv1.name_scope(None):   # clear any name scope it might get called in
            ret = tfv1.placeholder(
                tensor_spec.dtype, shape=tensor_spec.shape, name=tensor_spec.name)
        return ret


class InputDesc(
        namedtuple('InputDescTuple', ['type', 'shape', 'name'])):
    """
    An equivalent of `tf.TensorSpec`.

    History: this concept is used to represent metadata about the inputs,
    which can be later used to build placeholders or other types of input source.
    It is introduced much much earlier than the equivalent concept `tf.TensorSpec`
    was introduced in TensorFlow.
    Therefore, we now switched to use `tf.TensorSpec`, but keep this here for compatibility reasons.
    """

    def __new__(cls, type, shape, name):
        """
        Args:
            type (tf.DType):
            shape (tuple):
            name (str):
        """
        # TODO mark deprecated
        assert isinstance(type, tf.DType), type
        return tf.TensorSpec(shape=shape, dtype=type, name=name)


class ModelDescBase(object):
    """
    Base class for a model description.
    """

    @memoized_method
    def get_inputs_desc(self):
        # TODO mark deprecated
        return self.get_input_signature()

    @memoized_method
    def get_input_signature(self):
        """
        Returns:
            A list of :class:`tf.TensorSpec`, which describes the inputs of this model.
            The result is cached for each instance of :class:`ModelDescBase`.
        """
        with tf.Graph().as_default() as G:   # create these placeholder in a temporary graph
            inputs = self.inputs()
            assert isinstance(inputs, (list, tuple)), \
                "ModelDesc.inputs() should return a list of tf.TensorSpec objects! Got {} instead.".format(str(inputs))
            if isinstance(inputs[0], tf.Tensor):
                for p in inputs:
                    assert "Placeholder" in p.op.type, \
                        "inputs() have to return TensorSpec or placeholders! Found {} instead.".format(p)
                    assert p.graph == G, "Placeholders returned by inputs() should be created inside inputs()!"
            return [TensorSpec(shape=p.shape, dtype=p.dtype, name=get_op_tensor_name(p.name)[0]) for p in inputs]

    @property
    def input_names(self):
        """
        Returns:
            [str]: the names of all the inputs.
        """
        return [k.name for k in self.get_input_signature()]

    def inputs(self):
        """
        Returns a list of :class:`tf.TensorSpec` or placeholders.
        A subclass is expected to implement this method.

        If returning placeholders,
        the placeholders __have to__ be created inside this method.
        Don't return placeholders created in other places.

        Also, you should never call this method by yourself.

        Returns:
            list[tf.TensorSpec or tf.placeholder]. To be converted to :class:`tf.TensorSpec`.
        """
        raise NotImplementedError()

    def build_graph(self, *args):
        """
        Build the whole symbolic graph.
        This is supposed to be part of the "tower function" when used with :class:`TowerTrainer`.

        A subclass is expected to implement this method.

        Args:
            args ([tf.Tensor]): tensors that matches the list of inputs defined by ``inputs()``.

        Returns:
            In general it returns nothing, but a subclass
            may require it to return necessary information to build the trainer.
            For example, `SingleCostTrainer` expect this method to return the cost tensor.
        """
        raise NotImplementedError()


class ModelDesc(ModelDescBase):
    """
    A ModelDesc with **single cost** and **single optimizer**.
    It has the following constraints in addition to :class:`ModelDescBase`:

    1. :meth:`build_graph(...)` method should return a cost when called under a training context.
       The cost will be the final cost to be optimized by the optimizer.
       Therefore it should include necessary regularization.

    2. Subclass is expected to implement :meth:`optimizer()` method.

    """

    @memoized_method
    def get_optimizer(self):
        """
        Return the memoized optimizer returned by `optimizer()`.

        Users of :class:`ModelDesc` will need to implement `optimizer()`,
        which will only be called once per each model.

        Returns:
            a :class:`tf.train.Optimizer` instance.
        """
        ret = self.optimizer()
        assert isinstance(ret, tfv1.train.Optimizer), \
            "ModelDesc.optimizer() must return a tf.train.Optimizer! Got {} instead.".format(str(ret))
        return ret

    def optimizer(self):
        """
        Returns a `tf.train.Optimizer` instance.
        A subclass is expected to implement this method.
        """
        raise NotImplementedError()

    @deprecated("Just use `build_graph` instead!")
    def _build_graph_get_cost(self, *inputs):
        return self.build_graph(*inputs)
