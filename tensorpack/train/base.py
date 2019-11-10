# -*- coding: utf-8 -*-
# File: base.py

import copy
import time
import weakref
import tensorflow as tf

from ..compat import tfv1
from ..callbacks import Callback, Callbacks, Monitors, MonitorBase
from ..callbacks.steps import MaintainStepCounter
from ..tfutils import get_global_step_value
from ..tfutils.model_utils import describe_trainable_vars
from ..tfutils.sesscreate import NewSessionCreator, ReuseSessionCreator
from ..tfutils.sessinit import JustCurrentSession, SessionInit
from ..utils import logger
from ..utils.argtools import call_only_once
from ..utils.utils import humanize_time_delta
from .config import DEFAULT_CALLBACKS, DEFAULT_MONITORS, TrainConfig

__all__ = ['StopTraining', 'Trainer']


class StopTraining(Exception):
    """
    An exception thrown to stop training.
    """
    pass


class TrainLoop(object):
    """
    Manage the double for loop.
    """

    def __init__(self):
        self._epoch_num = 0
        self._global_step = 0
        self._local_step = -1

    def config(self, steps_per_epoch, starting_epoch, max_epoch):
        """
        Configure the loop given the settings.
        """
        self.starting_epoch = int(starting_epoch)
        self.max_epoch = int(max_epoch)
        self.steps_per_epoch = int(steps_per_epoch)
        # Allow empty epoch (no steps), if we want to run the callbacks only.
        assert self.steps_per_epoch >= 0 and self.max_epoch >= 0

        self._epoch_num = starting_epoch - 1

    def update_global_step(self):
        """
        Update the Python-side global_step from TF.
        This must be called under initialized default session.
        """
        self._global_step = get_global_step_value()

    @property
    def epoch_num(self):
        """
        The number of the currently ongoing epoch.

        An epoch is defined to cover the moment before calling `before_epoch` until after calling `trigger_epoch`.
        i.e., in the `trigger_epoch` of epoch 3, `self.epoch_num` is 3.
        If you need use `self.epoch_num` in your callback, you'll need to know this.
        """
        return self._epoch_num

    @property
    def global_step(self):
        """
        The tensorflow global_step, i.e. how many times ``hooked_sess.run`` has been called.

        Note:
            1. global_step is incremented **after** each ``hooked_sess.run`` returns from TF runtime.
            2. If you make zero or more than one calls to ``hooked_sess.run`` in one
               :meth:`run_step`, local_step and global_step may increment at different speed.
        """
        return self._global_step

    @property
    def local_step(self):
        """
        The number of steps that have finished in the current epoch.
        """
        return self._local_step


class Trainer(object):
    """ Base class for a trainer.
    """

    is_chief = True
    """
    Whether this process is the chief worker in distributed training.
    Certain callbacks will only be run by chief worker.
    """

    sess = None
    """
    The ``tf.Session`` object the trainer is using.
    Available after :meth:`initialize()`.

    Using ``trainer.sess.run`` to evaluate tensors that depend on the training
    ``InputSource`` may have unexpected effect:

    For example, if you use ``trainer.sess.run`` to evaluate a tensor that depends on the
    inputs coming from a ``StagingArea``,
    it will take a datapoint from the ``StagingArea``, making the ``StagingArea`` empty, and as a result
    make the training hang.
    """

    hooked_sess = None
    """
    The ``tf.train.MonitoredSession`` object the trainer is using.
    It contains all the ``before_run/after_run`` hooks the callbacks have registered.
    It is used for running the training iterations.
    Available after :meth:`initialize()`.

    Note that using ``hooked_sess.run`` will evaluate all the hooks,
    just like running a training iteration. It may do the following:

    1. Take a datapoint from the InputSource
    2. Increase the global_step
    3. Evaluate some summaries

    Typically you __should not__ use ``hooked_sess.run`` in callbacks,
    because it is for the "training iteration". If you just want to evaluate
    some tensors, use ``sess.run`` if the tensors does not depend on the inputs,
    or more generally, use `before_run/after_run` to evaluate the tensors **along with**
    the training iterations.
    """

    def __init__(self):
        self._callbacks = []
        self.loop = TrainLoop()

    def _register_callback(self, cb):
        """
        Register callbacks to the trainer.
        It can only be called before :meth:`Trainer.train()`.

        Args:
            cb (Callback or [Callback]): a callback or a list of callbacks

        Returns:
            succeed or not
        """
        if isinstance(cb, (list, tuple)):
            for x in cb:
                self._register_callback(x)
            return
        assert isinstance(cb, Callback), cb
        assert not isinstance(self._callbacks, Callbacks), \
            "Cannot register more callbacks after trainer was setup!"
        if not self.is_chief and cb.chief_only:
            logger.warn("Callback {} is chief-only, skipped.".format(str(cb)))
            return False
        else:
            self._callbacks.append(cb)
            return True

    register_callback = _register_callback

    def run_step(self):
        """
        Defines what to do in one iteration. The default is:
        ``self.hooked_sess.run(self.train_op)``.

        The behavior of each iteration can be changed by either setting ``trainer.train_op``,
        or overriding this method.
        """
        if not hasattr(self, 'train_op'):
            raise NotImplementedError(
                "Please either set `Trainer.train_op` or provide an implementation "
                "of Trainer.run_step()!")
        self.hooked_sess.run(self.train_op)

    @call_only_once
    def setup_callbacks(self, callbacks, monitors):
        """
        Setup callbacks and monitors. Must be called after the main graph is built.

        Args:
            callbacks ([Callback]):
            monitors ([MonitorBase]):
        """
        assert isinstance(callbacks, list), callbacks
        assert isinstance(monitors, list), monitors
        describe_trainable_vars()   # TODO weird

        self.register_callback(MaintainStepCounter())
        for cb in callbacks:
            self.register_callback(cb)
        for cb in self._callbacks:
            assert not isinstance(cb, MonitorBase), "Monitor cannot be pre-registered for now!"
        registered_monitors = []
        for m in monitors:
            if self.register_callback(m):
                registered_monitors.append(m)
        self.monitors = Monitors(registered_monitors)
        self.register_callback(self.monitors)   # monitors is also a callback

        # some final operations that might modify the graph
        logger.info("Setup callbacks graph ...")
        self._callbacks = Callbacks(self._callbacks)
        self._callbacks.setup_graph(weakref.proxy(self))

    @call_only_once
    def initialize(self, session_creator, session_init):
        """
        Create the session and set `self.sess`.
        Call `self.initiailize_hooks()`
        Finalize the graph.

        It must be called after callbacks are setup.

        Args:
            session_creator (tf.train.SessionCreator):
            session_init (sessinit.SessionInit):
        """
        assert isinstance(session_creator, tfv1.train.SessionCreator), session_creator
        assert isinstance(session_init, SessionInit), session_init
        session_init._setup_graph()

        logger.info("Creating the session ...")

        self.sess = session_creator.create_session()
        self.initialize_hooks()

        if self.is_chief:
            logger.info("Initializing the session ...")
            session_init._run_init(self.sess)
        else:
            if not isinstance(session_init, JustCurrentSession):
                logger.warn("This is not a chief worker, 'session_init' was ignored!")

        self.sess.graph.finalize()
        logger.info("Graph Finalized.")

    @call_only_once
    def initialize_hooks(self):
        """
        Create SessionRunHooks for all callbacks, and hook it onto `self.sess` to create `self.hooked_sess`.

        A new trainer may override this method to create multiple groups of hooks,
        which can be useful when the training is not done by a single `train_op`.
        """
        hooks = self._callbacks.get_hooks()
        self.hooked_sess = tfv1.train.MonitoredSession(
            session_creator=ReuseSessionCreator(self.sess), hooks=hooks)

    @call_only_once
    def main_loop(self, steps_per_epoch, starting_epoch, max_epoch):
        """
        Run the main training loop.

        Args:
            steps_per_epoch, starting_epoch, max_epoch (int):
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
                    self._callbacks.before_epoch()
                    start_time = time.time()
                    for self.loop._local_step in range(self.loop.steps_per_epoch):
                        if self.hooked_sess.should_stop():
                            return
                        self.run_step()  # implemented by subclass
                        self._callbacks.trigger_step()
                    self._callbacks.after_epoch()
                    logger.info("Epoch {} (global_step {}) finished, time:{}.".format(
                        self.loop.epoch_num, self.loop.global_step, humanize_time_delta(time.time() - start_time)))

                    # trigger epoch outside the timing region.
                    self._callbacks.trigger_epoch()
                logger.info("Training has finished!")
            except (StopTraining, tf.errors.OutOfRangeError) as e:
                logger.info("Training was stopped by exception {}.".format(str(e)))
            except KeyboardInterrupt:
                logger.info("Detected Ctrl-C and exiting main loop.")
                raise
            finally:
                self._callbacks.after_train()
                self.hooked_sess.close()

    def train(self,
              callbacks, monitors,
              session_creator, session_init,
              steps_per_epoch, starting_epoch=1, max_epoch=9999999):
        """
        Implemented by three lines:

        .. code-block:: python

            self.setup_callbacks(callbacks, monitors)
            self.initialize(session_creator, session_init)
            self.main_loop(steps_per_epoch, starting_epoch, max_epoch)

        You can call those methods by yourself to have better control on details if needed.
        """
        self.setup_callbacks(callbacks, monitors)
        self.initialize(session_creator, session_init)
        self.main_loop(steps_per_epoch, starting_epoch, max_epoch)

    def train_with_defaults(
            self, _sentinel=None,
            callbacks=None, monitors=None,
            session_creator=None, session_init=None,
            steps_per_epoch=None, starting_epoch=1, max_epoch=9999999,
            extra_callbacks=None):
        """
        Same as :meth:`train()`, except:

        1. Add `extra_callbacks` to callbacks. The default value for
           `extra_callbacks` is :meth:`DEFAULT_CALLBACKS()`.
        2. Default value for `monitors` is :meth:`DEFAULT_MONITORS()`.
        3. Provide default values for every option except `steps_per_epoch`.
        """
        assert _sentinel is None, "Please call `train_with_defaults` with keyword arguments only!"
        callbacks = copy.copy(callbacks or [])
        monitors = DEFAULT_MONITORS() if monitors is None else monitors
        extra_callbacks = DEFAULT_CALLBACKS() if extra_callbacks is None else extra_callbacks
        callbacks.extend(extra_callbacks)

        assert steps_per_epoch is not None
        session_creator = session_creator or NewSessionCreator()
        session_init = session_init or JustCurrentSession()

        self.train(callbacks, monitors,
                   session_creator, session_init,
                   steps_per_epoch, starting_epoch, max_epoch)

    def __new__(cls, *args, **kwargs):
        if (len(args) > 0 and isinstance(args[0], TrainConfig)) \
                or 'config' in kwargs:
            logger.error("You're calling new trainers with old trainer API!")
            logger.error("See https://github.com/tensorpack/tensorpack/issues/458 for more information.")
            import sys
            sys.exit(1)
        else:
            return super(Trainer, cls).__new__(cls)


def _get_property(name):
    """
    Delegate property to self.loop
    """
    ret = property(
        lambda self: getattr(self.loop, name))
    try:
        ret.__doc__ = getattr(TrainLoop, name).__doc__
    except AttributeError:
        pass
    return ret


for name in ['global_step', 'local_step', 'steps_per_epoch',
             'epoch_num', 'starting_epoch', 'max_epoch']:
    setattr(Trainer, name, _get_property(name))
