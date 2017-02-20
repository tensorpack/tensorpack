# -*- coding: UTF-8 -*-
# File: base.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import tensorflow as tf
from abc import ABCMeta, abstractmethod
import six
from ..tfutils.common import get_op_or_tensor_by_name, get_global_step_value

__all__ = ['Callback', 'ProxyCallback', 'CallbackFactory', 'Triggerable']


@six.add_metaclass(ABCMeta)
class Callback(object):
    """ Base class for all callbacks

    Attributes:
        epoch_num(int): the number of the current epoch.
        global_step(int): the number of global steps that have finished.
        trainer(Trainer): the trainer.
        graph(tf.Graph): the graph.

    Note:
        These attributes are available only after (and including)
        :meth:`_setup_graph`.

    .. document private functions
    .. automethod:: _setup_graph
    .. automethod:: _before_train
    .. automethod:: _before_run
    .. automethod:: _after_run
    .. automethod:: _trigger_step
    .. automethod:: _trigger_epoch
    .. automethod:: _after_train
    """

    def setup_graph(self, trainer):
        self._steps_per_epoch = trainer.config.steps_per_epoch
        self.trainer = trainer
        self.graph = tf.get_default_graph()
        with tf.name_scope(type(self).__name__):
            self._setup_graph()

    def _setup_graph(self):
        """
        Called before finalizing the graph.
        Override this method to setup the ops used in the callback.
        This is the same as ``tf.train.SessionRunHook.begin()``.
        """
        pass

    def before_train(self):
        self._starting_step = get_global_step_value()
        self._before_train()

    def _before_train(self):
        """
        Called right before the first iteration. The main difference to
        `setup_graph` is that at this point the graph is finalized and a
        default session is initialized.
        Override this method to, e.g. run some operations under the session.

        This is similar to ``tf.train.SessionRunHook.after_create_session()``, but different:
        it is called after the session is initialized by :class:`tfutils.SessionInit`.
        """
        pass

    def before_run(self, ctx):
        fetches = self._before_run(ctx)
        if fetches is None:
            return None
        if isinstance(fetches, tf.train.SessionRunArgs):
            return fetches

        # also support list of names
        assert isinstance(fetches, list), fetches
        ret = []
        for f in fetches:
            if isinstance(f, (tf.Tensor, tf.Operation)):
                ret.append(f)
            else:
                # warn about speed
                ret.append(get_op_or_tensor_by_name(f))
        return tf.train.SessionRunArgs(fetches=ret)

    def _before_run(self, ctx):
        """
        It is called before every ``hooked_sess.run()`` call, and it
        registers some extra op/tensors to run in the next call.
        This method is the same as ``tf.train.SessionRunHook.before_run``.
        Refer to TensorFlow docs for more details.

        An extra feature is that you can also simply return a list of names,
        instead of a ``tf.train.SessionRunArgs``.
        """
        return None

    def after_run(self, run_context, run_values):
        self._after_run(run_context, run_values)

    def _after_run(self, run_context, run_values):
        """
        It is called after every ``hooked_sess.run()`` call, and it
        processes the values requested by the corresponding :meth:`before_run`.
        It is equivalent to ``tf.train.SessionRunHook.after_run()``, refer to
        TensorFlow docs for more details.
        """
        pass

    def trigger_step(self):
        self._trigger_step()

    def _trigger_step(self):
        """
        Called after each :meth:`Trainer.run_step()` completes.

        You can override it to implement, e.g. a ProgressBar.
        """
        pass

    def trigger_epoch(self):
        self._trigger_epoch()

    def _trigger_epoch(self):
        """
        Called after the completion of every epoch.
        """
        pass

    def after_train(self):
        self._after_train()

    def _after_train(self):
        """
        Called after training.
        """
        pass

    @property
    def epoch_num(self):
        return self.trainer.epoch_num

    @property
    def global_step(self):
        return self.trainer.global_step

    def __str__(self):
        return type(self).__name__


@six.add_metaclass(ABCMeta)
class Triggerable(Callback):
    """
    Base class for "triggerable" callback. It has a method :meth:`Triggerable.trigger()`
    which can be called either inside an epoch or between epochs.
    Other higher-level wrappers will take the responsibility to determine **when**
    to call the trigger.

    If an triggerable is used as a callback directly (instead of under other
    higher-level wrapper to control the trigger), it will by default trigger after
    every epoch. This is mainly for backward-compatibility and convenience.

    .. document private functions
    .. automethod:: _trigger
    .. automethod:: _trigger_epoch
    """

    def trigger(self):
        self._trigger()

    @abstractmethod
    def _trigger(self):
        """
        Override this method to define what to trigger.
        Note that this method may be called both inside an epoch and after an epoch.
        """
        pass

    def _trigger_epoch(self):
        """ If a :class:`Triggerable` is used as a callback directly,
            the default behavior is to run the trigger every epoch."""
        self.trigger()


class ProxyCallback(Callback):
    """ A callback which proxy all methods to another callback.
        It's useful as a base class of callbacks which decorate other callbacks.
    """

    def __init__(self, cb):
        """
        Args:
            cb(Callback): the underlying callback
        """
        assert isinstance(cb, Callback), type(cb)
        self.cb = cb

    def _before_train(self):
        self.cb.before_train()

    def _setup_graph(self):
        self.cb.setup_graph(self.trainer)

    def _trigger_epoch(self):
        self.cb.trigger_epoch()

    def _trigger_step(self):
        self.cb.trigger_step()

    def _after_train(self):
        self.cb.after_train()

    def _before_run(self, ctx):
        self.cb._before_run(ctx)

    def _after_run(self, ctx, run_values):
        self.cb._after_run(ctx, run_values)

    def __str__(self):
        return "Proxy-" + str(self.cb)


class CallbackFactory(Callback):
    """
    Create a callback with some lambdas.
    """
    def __init__(self, setup_graph=None, before_train=None,
                 trigger_epoch=None, after_train=None):
        """
        Each lambda takes ``self`` as the only argument.
        """

        self._cb_setup_graph = setup_graph
        self._cb_before_train = before_train
        self._cb_trigger_epoch = trigger_epoch
        self._cb_after_train = after_train

    def _setup_graph(self):
        if self._cb_setup_graph:
            self._cb_setup_graph(self)

    def _before_train(self):
        if self._cb_before_train:
            self._cb_before_train(self)

    def _trigger_epoch(self):
        if self._cb_trigger_epoch:
            self._cb_trigger_epoch(self)

    def _after_train(self):
        if self._cb_after_train:
            self._cb_after_train(self)
