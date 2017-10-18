# -*- coding: utf-8 -*-
# File: base.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import time
import weakref
import six
from six.moves import range

import tensorflow as tf

from .config import TrainConfig
from ..utils import logger
from ..utils.develop import log_deprecated
from ..callbacks import Callback, Callbacks
from ..callbacks.monitor import Monitors, TrainingMonitor
from ..tfutils import get_global_step_value
from ..tfutils.model_utils import describe_trainable_vars
from ..tfutils.sesscreate import ReuseSessionCreator
from ..tfutils.sessinit import JustCurrentSession

from ..graph_builder.predictor_factory import PredictorFactory
from ..callbacks.steps import MaintainStepCounter

__all__ = ['Trainer', 'StopTraining']


class StopTraining(BaseException):
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
        self.starting_epoch = starting_epoch
        self.max_epoch = max_epoch
        self.steps_per_epoch = steps_per_epoch

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

    Attributes:
        config (TrainConfig): the config used in this trainer.
        model (ModelDesc): alias for ``config.model``.
        sess (tf.Session): the current session in use.
        hooked_sess (tf.train.MonitoredSession): the session with hooks.
        monitors (Monitors): the monitors. Other callbacks can use it for logging.
    """

    is_chief = True
    """
    Whether this process is the chief worker in distributed training.
    Only chief worker will run some callbacks.
    """

    def __init__(self, config):
        """
        Args:
            config (TrainConfig): the train config.
        """
        assert isinstance(config, TrainConfig), type(config)
        self._config = config
        self.model = config.model

        self._callbacks = []
        self._monitors = []
        self.loop = TrainLoop()
        self.loop.config(config.steps_per_epoch, config.starting_epoch, config.max_epoch)

        self._setup()   # subclass will setup the graph and InputSource

    def register_callback(self, cb):
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

    def register_monitor(self, mon):
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
            self._monitors.append(mon)
            self.register_callback(mon)

    @property
    def monitors(self):
        assert isinstance(self._monitors, Monitors), "Monitors haven't been setup!"
        return self._monitors

    def train(self):
        """ Start training """
        self.setup()
        self.main_loop()

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

    def setup(self):
        """
        Setup the trainer and be ready for the main loop.
        """
        self.register_callback(MaintainStepCounter())
        for cb in self._config.callbacks:
            self.register_callback(cb)
        for m in self._config.monitors:
            self.register_monitor(m)
        self._monitors = Monitors(self._monitors)
        self.register_callback(self._monitors)

        describe_trainable_vars()

        # some final operations that might modify the graph
        logger.info("Setup callbacks graph ...")
        self._callbacks = Callbacks(self._callbacks)
        self._callbacks.setup_graph(weakref.proxy(self))

        logger.info("Creating the session ...")
        self._create_session()

        if self.is_chief:
            logger.info("Initializing the session ...")
            self._config.session_init.init(self.sess)
        else:
            assert isinstance(self._config.session_init, JustCurrentSession), \
                "session_init is only valid for chief worker session!"

        self.sess.graph.finalize()
        logger.info("Graph Finalized.")

    def _create_session(self):
        """
        Setup self.sess (the raw tf.Session)
        and self.hooked_sess (the session with hooks and coordinator)
        """
        hooks = self._callbacks.get_hooks()
        self.sess = self._config.session_creator.create_session()
        self.hooked_sess = tf.train.MonitoredSession(
            session_creator=ReuseSessionCreator(self.sess), hooks=hooks)

    def _setup(self):
        """
        Build the entire graph for training.
        Responsible for setup InputSource as well (including registering InputSource callbacks)

        Since this method will get called in constructor only,
        you can simply leave it empty and build your graph outside the trainer.
        """
        pass

    def main_loop(self):
        """
        Run the main training loop.
        """
        with self.sess.as_default():
            self.loop.update_global_step()
            try:
                self._callbacks.before_train()
                # refresh global step (might have changed by callbacks) TODO ugly
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

    def get_predictor(self, input_names, output_names, tower=0):
        """
        Returns a callable predictor built under ``is_training=False`` tower context.
        Note that this method is only valid when this trainer has a ``ModelDesc``.

        Args:
            input_names (list), output_names(list): list of names
            tower (int): build the predictor on device '/gpu:{tower}' or use -1 for '/cpu:0'.

        Returns:
            an :class:`OnlinePredictor`.
        """
        # TODO move the logic to factory?
        return self.predictor_factory.get_predictor(input_names, output_names, tower)

    @property
    def predictor_factory(self):
        assert self.model is not None, \
            "Predictor can only be built one Trainer has ModelDesc!"
        if not hasattr(self, '_predictor_factory'):
            self._predictor_factory = PredictorFactory(
                self.model, self.vs_name_for_predictor)
        return self._predictor_factory

    @property
    def vs_name_for_predictor(self):
        # The vs name a predictor should be built under.
        # for internal use only. Should let graphbuilder return it.
        return ""

    @property
    def config(self):
        log_deprecated(
            "Trainer.config",
            "It is supposed to be private! Most of its attributes can be accessed by other means.",
            "2017-12-31")
        return self._config


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


def launch_train(
        run_step, model=None, callbacks=None, extra_callbacks=None, monitors=None,
        session_creator=None, session_config=None, session_init=None,
        starting_epoch=1, steps_per_epoch=None, max_epoch=99999):
    """
    ** Work In Progress! Don't use**

    This is another trainer interface, to start training **after** the graph has been built already.
    You can build the graph however you like
    (with or without tensorpack), and invoke this function to start training with callbacks & monitors.
    This provides the flexibility to define the training config after graph has been buit.
    One typical use is that callbacks often depend on names that are not known
    only until the graph has been built.

    Args:
        run_step (tf.Tensor or function): Define what the training iteration is.
            If given a Tensor/Operation, will eval it every step.
            If given a function, will invoke this function under the default session in every step.
        model (None or ModelDesc): Certain callbacks (e.g. InferenceRunner) depends on
            the existence of :class:`ModelDesc`. If you use a :class:`ModelDesc` to
            build the graph, add it here to to allow those callbacks to work.
            If you didn't use :class:`ModelDesc`, leave it empty.
        Other arguments are the same as in :class:`TrainConfig`.

    Examples:

    .. code-block:: python

        model = MyModelDesc()
        train_op, cbs = SimpleTrainer.setup_graph(model, QueueInput(mydataflow))
        launch_train(train_op, model=model, callbacks=[...] + cbs, steps_per_epoch=mydataflow.size())
        # the above is equivalent to:
        config = TrainConfig(model=MyModelDesc(), data=QueueInput(mydataflow) callbacks=[...])
        SimpleTrainer(config).train()
    """
    assert steps_per_epoch is not None, steps_per_epoch
    trainer = Trainer(TrainConfig(
        model=model,
        callbacks=callbacks,
        extra_callbacks=extra_callbacks,
        monitors=monitors,
        session_creator=session_creator,
        session_config=session_config,
        session_init=session_init,
        starting_epoch=starting_epoch,
        steps_per_epoch=steps_per_epoch,
        max_epoch=max_epoch))
    if isinstance(run_step, (tf.Tensor, tf.Operation)):
        trainer.train_op = run_step
    else:
        assert callable(run_step), run_step
        trainer.run_step = lambda self: run_step()
    trainer.train()
