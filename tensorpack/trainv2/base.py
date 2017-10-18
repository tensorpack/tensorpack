#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: base.py

import tensorflow as tf
import weakref
import time
from six.moves import range
import six
from abc import abstractmethod, ABCMeta

from ..utils import logger
from ..callbacks import Callback, Callbacks
from ..callbacks.monitor import Monitors, TrainingMonitor
from ..tfutils.model_utils import describe_trainable_vars
from ..tfutils.sessinit import JustCurrentSession
from ..tfutils.sesscreate import ReuseSessionCreator
from ..callbacks.steps import MaintainStepCounter

from ..train.base import StopTraining, TrainLoop

__all__ = ['Trainer', 'SingleCostTrainer']


class Trainer(object):
    """ Base class for a trainer.
    """

    is_chief = True

    def __init__(self):
        self._callbacks = []
        self.loop = TrainLoop()
        self._monitors = []  # Clarify the type. Don't change from list to monitors.

    def _register_callback(self, cb):
        """
        Register a callback to the trainer.
        It can only be called before :meth:`Trainer.train` gets called.
        """
        assert isinstance(cb, Callback), cb
        assert not isinstance(self._callbacks, Callbacks), \
            "Cannot register more callbacks after trainer was setup!"
        if not self.is_chief and cb.chief_only:
            logger.warn("Callback {} is chief-only, skipped.".format(str(cb)))
        else:
            self._callbacks.append(cb)

    def _register_monitor(self, mon):
        """
        Register a monitor to the trainer.
        It can only be called before :meth:`Trainer.train` gets called.
        """
        assert isinstance(mon, TrainingMonitor), mon
        assert not isinstance(self._monitors, Monitors), \
            "Cannot register more monitors after trainer was setup!"
        if not self.is_chief and mon.chief_only:
            logger.warn("Monitor {} is chief-only, skipped.".format(str(mon)))
        else:
            self._register_callback(mon)

    def run_step(self):
        """
        Defines what to do in one iteration. The default is:
        ``self.hooked_sess.run(self.train_op)``.

        The behavior can be changed by either defining what is ``train_op``,
        or overriding this method.
        """
        if not hasattr(self, 'train_op'):
            raise NotImplementedError(
                "Please either set `Trainer.train_op` or provide an implementation "
                "of Trainer.run_step()!")
        self.hooked_sess.run(self.train_op)

    def setup_callbacks(self, callbacks, monitors):
        """
        Setup callbacks and monitors. Must be called after the main graph is built.
        """
        describe_trainable_vars()   # TODO weird

        self._register_callback(MaintainStepCounter())
        for cb in callbacks:
            self._register_callback(cb)
        for m in monitors:
            self._register_monitor(m)
        self.monitors = Monitors(monitors)
        self._register_callback(self.monitors)   # monitors is also a callback

        # some final operations that might modify the graph
        logger.info("Setup callbacks graph ...")
        self._callbacks = Callbacks(self._callbacks)
        self._callbacks.setup_graph(weakref.proxy(self))

    def initialize(self, session_creator, session_init):
        """
        Initialize self.sess and self.hooked_sess.
        Must be called after callbacks are setup.
        """
        logger.info("Creating the session ...")

        hooks = self._callbacks.get_hooks()
        self.sess = session_creator.create_session()
        self.hooked_sess = tf.train.MonitoredSession(
            session_creator=ReuseSessionCreator(self.sess), hooks=hooks)

        if self.is_chief:
            logger.info("Initializing the session ...")
            session_init.init(self.sess)
        else:
            assert isinstance(session_init, JustCurrentSession), \
                "session_init is only valid for chief worker session!"

        self.sess.graph.finalize()
        logger.info("Graph Finalized.")

    def _create_session(self):
        """
        Setup self.sess (the raw tf.Session)
        and self.hooked_sess (the session with hooks and coordinator)
        """

    def main_loop(self, steps_per_epoch, starting_epoch=1, max_epoch=99999):
        """
        Run the main training loop.
        """
        with self.sess.as_default():
            self.loop.config(steps_per_epoch, starting_epoch, max_epoch)
            self.loop.update_global_step()
            try:
                self._callbacks.before_train()
                # refresh global step (might have changed by callbacks) TODO ugly
                # what if gs is changed later?
                self.loop.update_global_step()
                for self.loop._epoch_num in range(
                        self.loop.starting_epoch, self.loop.max_epoch + 1):
                    logger.info("Start Epoch {} ...".format(self.loop.epoch_num))
                    start_time = time.time()
                    self._callbacks.before_epoch()
                    for self.loop._local_step in range(self.loop.steps_per_epoch):
                        if self.hooked_sess.should_stop():
                            return
                        self.run_step()  # implemented by subclass
                        self._callbacks.trigger_step()
                    self._callbacks.after_epoch()
                    logger.info("Epoch {} (global_step {}) finished, time:{:.2f} sec.".format(
                        self.loop.epoch_num, self.loop.global_step, time.time() - start_time))

                    # trigger epoch outside the timing region.
                    self._callbacks.trigger_epoch()
                logger.info("Training has finished!")
            except (StopTraining, tf.errors.OutOfRangeError):
                logger.info("Training was stopped.")
            except KeyboardInterrupt:
                logger.info("Detected Ctrl-C and exiting main loop.")
            except:
                raise
            finally:
                self._callbacks.after_train()
                self.hooked_sess.close()

    def train(self,
              callbacks, monitors,
              session_creator, session_init,
              steps_per_epoch, starting_epoch, max_epoch):
        """
        Implemented by:

        .. code-block:: python

            self.setup_callbacks(callbacks, monitors)
            self.initialize(session_creator, session_init)
            self.main_loop(steps_per_epoch, starting_epoch, max_epoch)

        You can call those methods by yourself to have better control on details if needed.
        """
        self.setup_callbacks(callbacks, monitors)
        self.initialize(session_creator, session_init)
        self.main_loop(steps_per_epoch, starting_epoch, max_epoch)


def _get_property(name):
    """
    Delegate property to self.loop
    """
    ret = property(
        lambda self: getattr(self.loop, name))
    if six.PY3:     # __doc__ is readonly in Py2
        try:
            ret.__doc__ = getattr(TrainLoop, name).__doc__
        except AttributeError:
            pass
    return ret


for name in ['global_step', 'local_step', 'steps_per_epoch',
             'epoch_num', 'starting_epoch', 'max_epoch']:
    setattr(Trainer, name, _get_property(name))


@six.add_metaclass(ABCMeta)
class SingleCostTrainer(Trainer):
    """
    Base class for single-cost trainer.

    Single-cost trainer has a :meth:`setup_graph` method which takes
    (inputs_desc, input, get_cost_fn, get_opt_fn), and build the training operations from them.

    To use a SingleCostTrainer object, call `trainer.setup_graph(...); trainer.train(...)`.
    """

    def train(self,
              callbacks, monitors,
              session_creator, session_init,
              steps_per_epoch, starting_epoch, max_epoch):
        callbacks = callbacks + self._internal_callbacks
        Trainer.train(
            self,
            callbacks, monitors,
            session_creator, session_init,
            steps_per_epoch, starting_epoch, max_epoch)

    def setup_graph(self, inputs_desc, input, get_cost_fn, get_opt_fn):
        """
        Build the main training graph. Defaults to do nothing.
        You can either override it in subclasses, or build the graph outside
        the trainer.

        Returns:
            [Callback]: a (possibly empty) list of callbacks needed for training.
                These callbacks will be automatically added when you call `train()`.
                So you can usually ignore the return value.
        """
        assert not input.setup_done()
        input_callbacks = input.setup(inputs_desc)
        train_callbacks = self._setup_graph(input, get_cost_fn, get_opt_fn)
        self._internal_callbacks = input_callbacks + train_callbacks
        return self._internal_callbacks

    @abstractmethod
    def _setup_graph(self, input, get_cost_fn, get_opt_fn):
        pass
