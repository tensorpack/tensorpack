#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: tower.py

import tensorflow as tf
import six
from abc import abstractmethod, ABCMeta

from ..utils.argtools import call_only_once, memoized
from ..graph_builder.predict import SimplePredictBuilder
from ..input_source import PlaceholderInput
from ..predict.base import OnlinePredictor

from ..tfutils.tower import TowerFuncWrapper, get_current_tower_context
from ..tfutils.gradproc import FilterNoneGrad

from .base import Trainer

__all__ = ['SingleCostTrainer', 'TowerTrainer']


class TowerTrainer(Trainer):
    """
    Base trainers for models that can be built by calling a tower function under a :class:`TowerContext`.

    This is required by some features that replicates the model
    automatically, e.g. creating a predictor.
    """

    tower_func = None
    """
    A :class:`TowerFuncWrapper` instance.
    A callable which takes some input tensors and builds one replicate of the model.
    """

    @call_only_once
    def set_tower_func(self, tower_func):
        """
        Args:
            tower_func (TowerFuncWrapper)
        """
        assert isinstance(tower_func, TowerFuncWrapper), tower_func
        self.tower_func = tower_func

    @property
    def inputs_desc(self):
        """
        Returns:
            list[InputDesc]: metainfo about the inputs to the tower.
        """
        return self.tower_func.inputs_desc

    @property
    def towers(self):
        """
        Returns:
            a :class:`TowerTensorHandles` object, to
            access the tower handles by either indices or names.
        """
        return self.tower_func.towers

    def get_predictor(self, input_names, output_names, device=0):
        """
        Returns a callable predictor built under ``TowerContext(is_training=False)``.

        Args:
            input_names (list), output_names(list): list of names
            device (int): build the predictor on device '/gpu:{device}' or use -1 for '/cpu:0'.

        Returns:
            an :class:`OnlinePredictor`.
        """
        assert self.tower_func is not None, "Must set tower_func on the trainer to use get_predictor()!"
        tower_name = 'tower-pred-{}'.format(device) if device >= 0 else 'tower-pred-cpu'

        try:
            tower = self.tower_func.towers[tower_name]
        except KeyError:
            input = PlaceholderInput()
            input.setup(self.inputs_desc)

            with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                SimplePredictBuilder(
                    ns_name=tower_name, vs_name=self._main_tower_vs_name,
                    device=device).build(input, self.tower_func)
            tower = self.tower_func.towers[tower_name]
        input_tensors = tower.get_tensors(input_names)
        output_tensors = tower.get_tensors(output_names)
        return OnlinePredictor(input_tensors, output_tensors)

    @property
    def _main_tower_vs_name(self):
        """
        The vs name for the "main" copy of the model,
        to be used to build predictors.
        """
        return ""


@six.add_metaclass(ABCMeta)
class SingleCostTrainer(TowerTrainer):
    """
    Base class for single-cost trainer.

    Single-cost trainer has a :meth:`setup_graph` method which takes
    (inputs_desc, input, get_cost_fn, get_opt_fn), and build the training operations from them.

    To use a :class:`SingleCostTrainer` object, call `trainer.setup_graph(...); trainer.train(...)`.
    """

    @call_only_once
    def setup_graph(self, inputs_desc, input, get_cost_fn, get_opt_fn):
        """
        Responsible for building the main training graph for single-cost training.

        Args:
            inputs_desc ([InputDesc]):
            input (InputSource):
            get_cost_fn ([tf.Tensor] -> tf.Tensor): callable, takes some input tenosrs and return a cost tensor.
            get_opt_fn (-> tf.train.Optimizer): callable which returns an
                optimizer. Will only be called once.

        Note:
            1. `get_cost_fn` will always be called under a :class:`TowerContext`.
               which will contain information about reuse,
               training/inference, scope name, etc.
            2. `get_cost_fn` might get called multiple times for data-parallel training or inference.
            3. To respect variable reuse, use `tf.get_variable` instead of
               `tf.Variable` in `get_cost_fn`.
        """
        get_cost_fn = TowerFuncWrapper(get_cost_fn, inputs_desc)
        get_opt_fn = memoized(get_opt_fn)
        self.set_tower_func(get_cost_fn)

        # TODO setup may want to register monitor as well??
        input_callbacks = self._setup_input(inputs_desc, input)
        train_callbacks = self._setup_graph(input, get_cost_fn, get_opt_fn)
        internal_callbacks = input_callbacks + train_callbacks
        for cb in internal_callbacks:
            self.register_callback(cb)

    @abstractmethod
    def _setup_graph(self, input, get_cost_fn, get_opt_fn):
        """
        Implement the logic to build the graph, with an :class:`InputSource`
        that's been setup already.

        Returns:
            [Callback]: list of callbacks needed
        """

    def _setup_input(self, inputs_desc, input):
        assert not input.setup_done()
        return input.setup(inputs_desc)

    def _make_get_grad_fn(self, input, get_cost_fn, get_opt_fn):
        """
        Returns:
            a get_grad_fn for GraphBuilder to use.
        """
        # internal use only
        assert input.setup_done()

        def get_grad_fn():
            ctx = get_current_tower_context()
            cost = get_cost_fn(*input.get_input_tensors())

            if ctx.has_own_variables:
                varlist = ctx.get_collection_in_tower(tf.GraphKeys.TRAINABLE_VARIABLES)
            else:
                varlist = tf.trainable_variables()
            opt = get_opt_fn()
            grads = opt.compute_gradients(
                cost, var_list=varlist,
                gate_gradients=False, colocate_gradients_with_ops=True)
            grads = FilterNoneGrad().process(grads)
            return grads

        return get_grad_fn
