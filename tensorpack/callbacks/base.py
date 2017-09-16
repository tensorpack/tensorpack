# -*- coding: UTF-8 -*-
# File: base.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import tensorflow as tf
from abc import ABCMeta
import six
from ..utils.develop import log_deprecated
from ..tfutils.common import get_op_or_tensor_by_name

__all__ = ['Callback', 'ProxyCallback', 'CallbackFactory', 'Triggerable']


@six.add_metaclass(ABCMeta)
class Callback(object):
    """ Base class for all callbacks. See
    `Write a Callback
    <http://tensorpack.readthedocs.io/en/latest/tutorial/extend/callback.html>`_
    for more detailed explanation of the callback methods.

    Attributes:
        epoch_num(int): trainer.epoch_num
        global_step(int): trainer.global_step
        local_step(int): trainer.local_step
        trainer(Trainer): the trainer.
        graph(tf.Graph): the graph.

    Note:
        These attributes are available only after (and including)
        :meth:`_setup_graph`.

    .. document private functions
    .. automethod:: _setup_graph
    .. automethod:: _before_train
    .. automethod:: _after_train
    .. automethod:: _before_run
    .. automethod:: _after_run
    .. automethod:: _before_epoch
    .. automethod:: _after_epoch
    .. automethod:: _trigger_step
    .. automethod:: _trigger_epoch
    .. automethod:: _trigger
    """

    _chief_only = True

    def setup_graph(self, trainer):
        self._steps_per_epoch = trainer.config.steps_per_epoch
        self.trainer = trainer
        self.graph = tf.get_default_graph()
        scope_name = type(self).__name__
        scope_name = scope_name.replace('_', '')
        with tf.name_scope(scope_name):
            self._setup_graph()

    def _setup_graph(self):
        """
        Called before finalizing the graph.
        Override this method to setup the ops used in the callback.
        This is the same as ``tf.train.SessionRunHook.begin()``.
        """
        pass

    def before_train(self):
        self._before_train()

    def _before_train(self):
        """
        Called right before the first iteration. The main difference to
        `setup_graph` is that at this point the graph is finalized and a default session is initialized.
        Override this method to, e.g. run some operations under the session.

        This is similar to ``tf.train.SessionRunHook.after_create_session()``, but different:
        it is called after the session is initialized by :class:`tfutils.SessionInit`.
        """
        pass

    def before_epoch(self):
        self._before_epoch()

    def _before_epoch(self):
        """
        Called right before each epoch.
        Usually you should use the :meth:`trigger` callback to run something between epochs.
        Use this method only when something really needs to be run **immediately** before each epoch.
        """
        pass

    def after_epoch(self):
        self._after_epoch()

    def _after_epoch(self):
        """
        Called right after each epoch.
        Usually you should use the :meth:`trigger` callback to run something between epochs.
        Use this method only when something really needs to be run **immediately** after each epoch.
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
        Called after each :meth:`Trainer.run_step()` completes. Defaults to no-op.

        You can override it to implement, e.g. a ProgressBar.
        """
        pass

    def trigger_epoch(self):
        self._trigger_epoch()

    def _trigger_epoch(self):
        """
        Called after the completion of every epoch. Defaults to call ``self.trigger()``
        """
        self.trigger()

    def trigger(self):
        self._trigger()

    def _trigger(self):
        """
        Override this method to define a general trigger behavior, to be used with trigger schedulers.
        Note that the schedulers (e.g. :class:`PeriodicTrigger`) might call this
        method both inside an epoch and after an epoch.

        When used without the scheduler, this method by default will be called by `trigger_epoch()`.
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

    @property
    def local_step(self):
        return self.trainer.local_step

    @property
    def chief_only(self):
        """
        Only run this callback on chief training process.

        Returns: bool
        """
        return self._chief_only

    @chief_only.setter
    def chief_only(self, v):
        self._chief_only = v

    def __str__(self):
        return type(self).__name__


# back-compat. in case someone write something in triggerable
Triggerable = Callback


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
        with tf.name_scope(None):
            self.cb.setup_graph(self.trainer)

    def _trigger_epoch(self):
        self.cb.trigger_epoch()

    def _trigger(self):
        self.cb.trigger()

    def _trigger_step(self):
        self.cb.trigger_step()

    def _after_train(self):
        self.cb.after_train()

    def _before_epoch(self):
        self.cb.before_epoch()

    def _after_epoch(self):
        self.cb.after_epoch()

    def _before_run(self, ctx):
        return self.cb._before_run(ctx)

    def _after_run(self, ctx, run_values):
        self.cb._after_run(ctx, run_values)

    def __str__(self):
        return "Proxy-" + str(self.cb)


class CallbackFactory(Callback):
    """
    Create a callback with some lambdas.
    """
    def __init__(self, setup_graph=None, before_train=None, trigger=None,
                 after_train=None, trigger_epoch=None):
        """
        Each lambda takes ``self`` as the only argument.

        Note:
            trigger_epoch was deprecated.
        """

        self._cb_setup_graph = setup_graph
        self._cb_before_train = before_train
        self._cb_trigger = trigger
        self._cb_after_train = after_train

        if trigger_epoch:
            self._cb_trigger = trigger_epoch
            log_deprecated("CallbackFactory(trigger_epoch=)", "Use trigger instead.", "2017-11-15")

    def _setup_graph(self):
        if self._cb_setup_graph:
            self._cb_setup_graph(self)

    def _before_train(self):
        if self._cb_before_train:
            self._cb_before_train(self)

    def _trigger(self):
        if self._cb_trigger:
            self._cb_trigger(self)

    def _after_train(self):
        if self._cb_after_train:
            self._cb_after_train(self)
