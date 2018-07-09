# -*- coding: utf-8 -*-
# File: model_desc.py


from collections import namedtuple
import tensorflow as tf

from ..utils import logger
from ..utils.argtools import memoized
from ..utils.develop import log_deprecated
from ..tfutils.tower import get_current_tower_context
from ..input_source import InputSource
from ..models.regularize import regularize_cost_from_collection

__all__ = ['InputDesc', 'ModelDesc', 'ModelDescBase']


class InputDesc(
        namedtuple('InputDescTuple', ['type', 'shape', 'name'])):
    """
    Metadata about an input entry point to the graph.
    This metadata can be later used to build placeholders or other types of
    input source.
    """

    def __new__(cls, type, shape, name):
        """
        Args:
            type (tf.DType):
            shape (tuple):
            name (str):
        """
        shape = tuple(shape)    # has to be tuple for "self" to be hashable
        assert isinstance(type, tf.DType), type
        if any(k in name for k in [':', '/', ' ']):
            raise ValueError("Invalid InputDesc name: '{}'".format(name))
        self = super(InputDesc, cls).__new__(cls, type, shape, name)
        self._cached_placeholder = {}
        return self

    # TODO this method seems unused outside this class
    def build_placeholder(self):
        """
        Build a tf.placeholder from the metadata.

        Returns:
            tf.Tensor:
        """
        with tf.name_scope(None):   # clear any name scope it might get called in
            ret = tf.placeholder(
                self.type, shape=self.shape, name=self.name)
        self._register_cached_placeholder(ret)
        return ret

    # cannot memoize here, because InputDesc is hashed by its fields.
    def build_placeholder_reuse(self):
        """
        Build a tf.placeholder from the metadata, or return an old one.

        Returns:
            tf.Tensor:
        """
        g = tf.get_default_graph()
        if g in self._cached_placeholder:
            return self._cached_placeholder[g]
        else:
            return self.build_placeholder()

    def _register_cached_placeholder(self, placeholder):
        graph = placeholder.graph
        assert graph not in self._cached_placeholder, \
            "Placeholder for this InputDesc had been created before! This is a bug."
        self._cached_placeholder[graph] = placeholder

    @staticmethod
    def from_placeholder(placeholder):
        name = placeholder.op.name
        if name.endswith('_1') or name.endswith('_2'):
            logger.error("Creating InputDesc from a placeholder named {}.".format(name))
            logger.error("You might have mistakenly created this placeholder multiple times!")
        ret = InputDesc(
            placeholder.dtype,
            tuple(placeholder.shape.as_list()),
            name)
        ret._register_cached_placeholder(placeholder)
        return ret


class ModelDescBase(object):
    """
    Base class for a model description.
    """

    @memoized
    def get_inputs_desc(self):
        """
        Returns:
            a list of :class:`InputDesc`.
        """
        try:
            return self._get_inputs()
        except NotImplementedError:
            with tf.Graph().as_default() as G:   # create these placeholder in a temporary graph
                inputs = self.inputs()
                for p in inputs:
                    assert p.graph == G, "Placeholders returned by inputs() should be created inside inputs()!"
                return [InputDesc.from_placeholder(p) for p in inputs]

    def _get_inputs(self):
        raise NotImplementedError()

    def inputs(self):
        """
        __Create__ and returns a list of placeholders.
        A subclass is expected to implement this method.

        The placeholders __have to__ be created inside this method.
        Don't return placeholders created in other methods.
        Also, you should not call this method by yourself.

        Returns:
            a list of `tf.placeholder`, to be converted to :class:`InputDesc`.
        """
        raise NotImplementedError()

    def build_graph(self, *args):
        """
        Build the whole symbolic graph.
        This is supposed to be part of the "tower function" when used with :class:`TowerTrainer`.

        A subclass is expected to overwrite this method.

        Args:
            args ([tf.Tensor]): tensors that matches the list of inputs defined by ``inputs()``.

        Returns:
            In general it returns nothing, but a subclass
            may require it to return necessary information to build the trainer.
            For example, `SingleCostTrainer` expect this method to return the cost tensor.
        """
        if len(args) == 1:
            arg = args[0]
            if isinstance(arg, InputSource):
                inputs = arg.get_input_tensors()  # remove in the future?
                log_deprecated("build_graph(InputSource)",
                               "Call with tensors in positional args instead.", "2018-03-31")
            elif isinstance(arg, (list, tuple)):
                inputs = arg
                log_deprecated("build_graph([Tensor])", "Call with positional args instead.", "2018-03-31")
            else:
                inputs = [arg]
        else:
            inputs = args

        assert len(inputs) == len(self.get_inputs_desc()), \
            "Number of inputs passed to the graph != number of inputs defined " \
            "in ModelDesc! ({} != {})".format(len(inputs), len(self.get_inputs_desc()))
        return self._build_graph(inputs)

    def _build_graph(self, inputs):
        """
        This is an alternative interface which takes a list of tensors, instead of positional arguments.
        By default :meth:`build_graph` will call this method.
        """
        pass


class ModelDesc(ModelDescBase):
    """
    A ModelDesc with **single cost** and **single optimizer**.
    It has the following constraints in addition to :class:`ModelDescBase`:

    1. :meth:`build_graph(...)` method should return a cost when called under a training context.
       The cost will be the final cost to be optimized by the optimizer.
       Therefore it should include necessary regularization.

    2. Subclass is expected to implement :meth:`optimizer()` method.

    """

    def get_cost(self):
        """
        Being deprecated.
        You're recommended to return a cost tensor in :meth:`build_graph` method directly.

        This function takes the `self.cost` tensor defined by :meth:`build_graph`,
        and applies the collection
        ``tf.GraphKeys.REGULARIZATION_LOSSES`` to the cost automatically.
        """
        cost = self._get_cost()
        reg_cost = regularize_cost_from_collection()
        if reg_cost.op.type != 'Const':
            logger.warn("Regularization losses found in collection, and a 'cost' tensor was "
                        "not returned by `build_graph`. Therefore applying regularization automatically!")
            return tf.add(cost, reg_cost, name='cost_with_regularizer')
        else:
            return cost

    def _get_cost(self, *args):
        return self.cost

    @memoized
    def get_optimizer(self):
        """
        Return the memoized optimizer returned by `optimizer()`.

        Users of :class:`ModelDesc` will need to implement `optimizer()`,
        which will only be called once per each model.

        Returns:
            a :class:`tf.train.Optimizer` instance.
        """
        try:
            return self._get_optimizer()
        except NotImplementedError:
            pass
        return self.optimizer()

    def _get_optimizer(self):
        raise NotImplementedError()

    def optimizer(self):
        """
        Returns a `tf.train.Optimizer` instance.
        A subclass is expected to implement this method.
        """
        raise NotImplementedError()

    def _build_graph_get_cost(self, *inputs):
        """
        Used internally by trainers to get the final cost for optimization in a backward-compatible way.
        """
        ret = self.build_graph(*inputs)
        if not get_current_tower_context().is_training:
            return None     # this is the tower function, could be called for inference
        if isinstance(ret, tf.Tensor):  # the preferred way
            return ret
        else:   # the old way, for compatibility
            return self.get_cost()
