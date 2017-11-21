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
from ..tfutils.tower import TowerFuncWrapper

from ..input_source import PlaceholderInput
from ..graph_builder.predict import SimplePredictBuilder
from ..predict.base import OnlinePredictor
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

    _API_VERSION = 1

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
        if self.model is not None:

            def f(*inputs):
                self.model.build_graph(*inputs)

            """
            Only to mimic new trainer interafce on inference.
            """
            self.inputs_desc = self.model.get_inputs_desc()
            self.tower_func = TowerFuncWrapper(f, self.inputs_desc)

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
        self._config.session_init._setup_graph()

        logger.info("Creating the session ...")
        self._create_session()

        if self.is_chief:
            logger.info("Initializing the session ...")
            self._config.session_init._run_init(self.sess)
        else:
            if not isinstance(self._config.session_init, JustCurrentSession):
                logger.warn("This is not a chief worker, 'session_init' was ignored!")

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
                raise
            finally:
                self._callbacks.after_train()
                self.hooked_sess.close()

    def get_predictor(self, input_names, output_names, tower=0):
        """
        Returns a callable predictor built under ``TowerContext(is_training=False)``.

        Args:
            input_names (list), output_names(list): list of names
            tower (int): build the predictor on device '/gpu:{tower}' or use -1 for '/cpu:0'.

        Returns:
            an :class:`OnlinePredictor`.
        """
        device = tower
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

    # create new trainer when not called with TrainConfig
    def __new__(cls, *args, **kwargs):
        if (len(args) > 0 and isinstance(args[0], TrainConfig)) \
                or 'config' in kwargs:
            return super(Trainer, cls).__new__(cls)
        else:
            import tensorpack.train as new_train
            name = cls.__name__
            new_trainer = getattr(new_train, name)
            logger.warn("You're calling old trainers with new trainer API!")
            logger.warn("Now it returns the new trainer for you, please `export TENSORPACK_TRAIN_API=v2`"
                        " to import new trainers automatically.")
            logger.warn("You can also ignore this warning and wait for new API to become the default.")
            return new_trainer(*args, **kwargs)


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
