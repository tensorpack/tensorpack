# -*- coding: utf-8 -*-
# File: tower.py


import tensorflow as tf
import six
from six.moves import zip
from abc import abstractproperty, abstractmethod, ABCMeta

from ..utils import logger
from ..utils.argtools import call_only_once
from ..utils.naming import MOVING_SUMMARY_OPS_KEY
from ..utils.develop import HIDE_DOC
from .collection import CollectionGuard
from .common import get_op_or_tensor_by_name, get_op_tensor_name

__all__ = ['get_current_tower_context', 'BaseTowerContext', 'TowerContext', 'TowerFuncWrapper',
           'TowerTensorHandle', 'TowerTensorHandles']

_CurrentTowerContext = None


@six.add_metaclass(ABCMeta)
class BaseTowerContext(object):
    """ A context where the current model is built in.
        You need to use :func:`TowerContext` to create a :class:`BaseTowerContext`.
    """

    @HIDE_DOC
    def __init__(self, ns_name, vs_name=''):
        """
        This is not supposed to be used by users.
        You need to use :func:`TowerContext` to create a :class:`BaseTowerContext`.

        Args:
            ns_name (str): The name scope of the tower.
            vs_name (str): Open a new variable scope with this name.
        """
        self._name = ns_name

        self._vs_name = vs_name
        if len(vs_name):
            assert len(ns_name), "TowerContext(vs_name) cannot be used with an empty name!"

    @abstractproperty
    def is_main_training_tower(self):
        """
        Whether this tower is the main (i.e., the first) training tower.
        """
        pass

    @abstractproperty
    def has_own_variables(self):
        """
        Whether this tower is supposed to have its own trainable variables.
        """
        pass

    @property
    def name(self):
        """
        Returns:
            str - The name scope of the tower.
        """
        return self._name

    @property
    def vs_name(self):
        """
        Returns:
            str - The variable scope of the tower.
        """
        return self._vs_name

    @property
    def ns_name(self):
        """
        Returns:
            str - The name scope of the tower.
        """
        return self._name

    def get_collection_in_tower(self, key):
        """
        From a collection, get items that are __added__ to the collection in this tower.

        Note that it works by tracking the collection at the beginning and end of
        the tower function.
        Therefore it does not guarantee that the items are __created__ in this tower.
        """
        return self._collection_guard.get_collection_in_tower(key)

    @call_only_once
    def _get_scopes(self):
        """
        Returns the ns and vs for this tower.
        """
        if not len(self._name):
            # work around https://github.com/tensorflow/tensorflow/issues/14703
            return [tf.variable_scope(tf.get_variable_scope())]

        ret = []

        if len(self._vs_name):
            ret.append(tf.variable_scope(self._vs_name))
        else:
            # caller should have handled reuse outside of TowerContext
            ret.append(tf.variable_scope(tf.get_variable_scope()))

        # always clear existing ns  # TODO check existing ns
        if len(self._name) and self._name != self._vs_name:
            ret.append(tf.name_scope(self._name + '/'))
        return ret

    @abstractmethod
    def _keys_to_freeze(self):
        pass

    def __enter__(self):
        global _CurrentTowerContext
        assert _CurrentTowerContext is None, "Cannot nest TowerContext!"
        _CurrentTowerContext = self

        self._collection_guard = CollectionGuard(
            self._name,
            check_diff=not self.is_main_training_tower,
            freeze_keys=self._keys_to_freeze())

        self._ctxs = self._get_scopes()
        self._ctxs.append(self._collection_guard)
        for c in self._ctxs:
            c.__enter__()

        # check that ns_name is always the same as _name
        ns = tf.get_default_graph().get_name_scope()
        assert ns == self._name, \
            "Name conflict: name_scope inside tower '{}' becomes '{}'!".format(self._name, ns) \
            + " You may need a different name for the tower!"
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        global _CurrentTowerContext
        _CurrentTowerContext = None

        if not self.has_own_variables:
            diff_trainable_vars = self._collection_guard.get_collection_in_tower(tf.GraphKeys.TRAINABLE_VARIABLES)
            assert len(diff_trainable_vars) == 0,  \
                "New TRAINABLE_VARIABLES shouldn't be created in {}: ".format(
                    self._name) + ', '.join([k.name for k in diff_trainable_vars])
        for c in self._ctxs[::-1]:
            c.__exit__(exc_type, exc_val, exc_tb)
        return False

    def __str__(self):
        return "TowerContext(name={}, is_training={})".format(
            self._name, self._is_training)


class TrainTowerContext(BaseTowerContext):

    is_training = True

    def __init__(self, ns_name, vs_name='', index=0, total=1):
        """
        Args:
            index (int): index of this tower, only used in training.
            total (int): total number of towers to be built.
        """
        super(TrainTowerContext, self).__init__(ns_name, vs_name)

        self.index = int(index)
        self.total = int(total)
        if self.index > 0:
            assert self.total > self.index, "(index, total) = ({}, {})".format(self.index, self.total)

        vs = tf.get_variable_scope()
        assert vs.name == '', "Cannot nest TrainTowerContext with an existing variable scope!"
        if self.has_own_variables:
            assert not vs.reuse, \
                "Cannot create tower {} under reuse=True!".format(ns_name)

    @property
    def is_main_training_tower(self):
        return self.index == 0

    @property
    def has_own_variables(self):
        return self.index == 0 or len(self._vs_name) > 0

    def _keys_to_freeze(self):
        if self.index == 0:
            return []
        return [tf.GraphKeys.SUMMARIES, MOVING_SUMMARY_OPS_KEY]


class PredictTowerContext(BaseTowerContext):

    is_training = False

    def __init__(self, ns_name, vs_name=''):
        super(PredictTowerContext, self).__init__(ns_name, vs_name)

        self._initial_vs_reuse = tf.get_variable_scope().reuse

    @property
    def has_own_variables(self):
        return not self._initial_vs_reuse

    @property
    def is_main_training_tower(self):
        return False

    def _keys_to_freeze(self):
        # freeze UPDATE_OPS during inference because they should never be used
        return [tf.GraphKeys.SUMMARIES, MOVING_SUMMARY_OPS_KEY, tf.GraphKeys.UPDATE_OPS]


def get_current_tower_context():
    """
    When called inside a TowerContext, returns the TowerContext.

    Returns:
        a :class:`BaseTowerContext` instance or None, if not called under a TowerContext.
    """
    return _CurrentTowerContext


def TowerContext(tower_name, is_training, vs_name=''):
    """
    The context for a tower function, containing metadata about the current tower.
    Tensorpack trainers use :class:`TowerContext` to manage tower function.
    Many tensorpack layers have to be called under a :class:`TowerContext`.

    Example:

    .. code-block:: python

        with TowerContext('', is_training=True):
            # call a tensorpack layer or a tower function
    """
    if is_training:
        return TrainTowerContext(tower_name, vs_name=vs_name)
    else:
        return PredictTowerContext(tower_name, vs_name=vs_name)


class TowerFuncWrapper(object):
    """
    A wrapper around a tower function (see
    [tutorial on tower function](http://tensorpack.readthedocs.io/tutorial/trainer.html#tower-trainer)).
    It keeps track of the name scope, variable scope and input/output tensors
    each time the function is called.

    :class:`TowerTrainer` needs this so that it knows how to build a predictor.
    """

    def __init__(self, tower_fn, inputs_desc):
        """
        Args:
            tower_func: a function which builds one tower in the graph.
                It takes several input tensors and could return anything.
            inputs_desc ([InputDesc]): list of :class:`InputDesc`.
                They are used to figure out the names for the input tensors.
        """
        assert callable(tower_fn), tower_fn
        self._inputs_desc_names = [k.name for k in inputs_desc]
        assert len(set(self._inputs_desc_names)) == len(self._inputs_desc_names), \
            "Duplicated names in inputs_desc! " + str(self._inputs_desc_names)
        self._tower_fn = tower_fn
        self._inputs_desc = inputs_desc

        self._handles = []

    def __new__(cls, tower_fn, inputs_desc):
        # to avoid double-wrapping a function
        if isinstance(tower_fn, TowerFuncWrapper):
            return tower_fn
        else:
            return super(TowerFuncWrapper, cls).__new__(cls)

    def __call__(self, *args):
        ctx = get_current_tower_context()
        assert ctx is not None, "Function must be called under TowerContext!"
        output = self._tower_fn(*args)
        handle = TowerTensorHandle(ctx, args, output, self._inputs_desc)
        self._handles.append(handle)
        return output

    @property
    def towers(self):
        """
        Returns:
            a :class:`TowerTensorHandles` object, that can
            access the tower handles by either indices or names.
        """
        return TowerTensorHandles(self._handles)

    @property
    def inputs_desc(self):
        return self._inputs_desc


class TowerTensorHandles(object):
    """
    Wrap a list of :class:`TowerTensorHandle`,
    to support access to them by index or names.
    """
    def __init__(self, handles):
        self._handles = handles
        self._name_to_handle = {k.ns_name: k for k in handles}

    def __len__(self):
        return len(self._handles)

    def __getitem__(self, name_or_index):
        """
        Args:
            name_or_index (str or int):

        Returns:
            a :class:`TowerTensorHandle`.
        """
        if isinstance(name_or_index, int):
            return self._handles[name_or_index]
        return self._name_to_handle[name_or_index]

    def training(self):
        """
        Returns:
            A :class:`TowerTensorHandles`, containing only the training towers.
        """
        handles = [h for h in self._handles if h.is_training]
        return TowerTensorHandles(handles)

    def inference(self):
        """
        Returns:
            A :class:`TowerTensorHandles`, containing only the inference towers.
        """
        handles = [h for h in self._handles if not h.is_training]
        return TowerTensorHandles(handles)


class TowerTensorHandle(object):
    """
    When a function is called multiple times under each tower,
    it becomes hard to keep track of the scope and access those tensors
    in each tower.
    This class provides easy access to the tensors as well as the
    inputs/outputs created in each tower.
    """

    @HIDE_DOC
    def __init__(self, ctx, input, output, inputs_desc=None):
        self._ctx = ctx

        self._extra_tensor_names = {}
        if inputs_desc is not None:
            assert len(inputs_desc) == len(input)
            self._extra_tensor_names = {
                get_op_tensor_name(x.name)[1]: y for x, y in zip(inputs_desc, input)}
        self._input = input
        self._output = output

    @property
    def vs_name(self):
        return self._ctx.vs_name

    @property
    def ns_name(self):
        return self._ctx.ns_name

    def get_tensor(self, name):
        """
        Get a tensor in this tower. The name can be:

        1. The name of the tensor without any tower prefix.

        2. The name of an :class:`InputDesc`, if it is used when building the tower.
        """
        name = get_op_tensor_name(name)[1]
        if len(self.ns_name):
            name_with_ns = self.ns_name + "/" + name
        else:
            name_with_ns = name

        try:
            ret = get_op_or_tensor_by_name(name_with_ns)
        except KeyError:
            if name in self._extra_tensor_names:
                return self._extra_tensor_names[name]
            raise
        else:
            if name in self._extra_tensor_names:
                logger.warn(
                    "'{}' may refer to both the tensor '{}' or the input '{}'.".format(
                        name, ret.name, self._extra_tensor_names[name].name) +
                    "Assuming it is the tensor '{}'.".format(ret.name))
            return ret

    def get_tensors(self, names):
        """
        Like :meth:`get_tensor`, but takes a list and returns a list.
        """
        return [self.get_tensor(name) for name in names]

    def __getitem__(self, name):
        """
        The same as :meth:`get_tensor`.
        """
        return self.get_tensor(name)

    def get_variable(self, name):
        """
        Get a variable used in this tower.
        The name should not contain the variable scope prefix of the tower.

        When the tower has the same variable scope and name scope, this is equivalent to
        :meth:`get_tensor`.
        """
        name = get_op_tensor_name(name)[1]
        if len(self.vs_name):
            name_with_vs = self.vs_name + "/" + name
        else:
            name_with_vs = name
        return get_op_or_tensor_by_name(name_with_vs)

    def get_variables(self, names):
        """
        Like :meth:`get_variable`, but takes a list and returns a list.
        """
        return [self.get_variable(name) for name in names]

    def get_collection(self, key=None, name=None):
        """
        See :meth:`BaseTowerContext.get_collection_in_tower`.

        Args:
            key (str): the key of the collection
            name: deprecated
        """
        if name is not None:
            logger.warn("TowerTensorHandle.get_collection(name=..) was renamed to (key=..) !")
            key = name
        return self._ctx.get_collection_in_tower(key)

    @property
    def input(self):
        """
        The list of input tensors used to build the tower.
        """
        return self._input

    @property
    def output(self):
        """
        The output returned by the tower function.
        """
        return self._output

    @property
    def is_training(self):
        return self._ctx.is_training
