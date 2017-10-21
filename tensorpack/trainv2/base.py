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
from ..utils.argtools import call_only_once, memoized
from ..callbacks import Callback, Callbacks
from ..callbacks.monitor import Monitors, TrainingMonitor
from ..tfutils.model_utils import describe_trainable_vars
from ..tfutils.sessinit import JustCurrentSession
from ..tfutils.sesscreate import ReuseSessionCreator
from ..tfutils.tower import TowerFuncWrapper, get_current_tower_context
from ..tfutils.gradproc import FilterNoneGrad
from ..callbacks.steps import MaintainStepCounter

from ..graph_builder.predictor_factory import SimplePredictBuilder
from ..input_source import FeedfreeInput, PlaceholderInput
from ..predict.base import OnlinePredictor

import tensorpack.train as old_train    # noqa
from ..train.base import StopTraining, TrainLoop

__all__ = ['Trainer', 'SingleCostTrainer', 'TowerTrainer']


class Trainer(object):
    """ Base class for a trainer.
    """

    _API_VERSION = 2

    is_chief = True

    def __init__(self, config=None):
        """
        config is only for compatibility reasons in case you're
        using custom trainers with old-style API.
        You should never use config.
        """
        self._callbacks = []
        self.loop = TrainLoop()
        self._monitors = []  # Clarify the type. Don't change from list to monitors.

        # Hacks!
        if config is not None:
            logger.warn("You're initializing new trainer with old trainer API!")
            logger.warn("This could happen if you wrote a custom trainer before.")
            logger.warn("It may work now through some hacks, but please switch to the new API!")
            self._config = config
            self.inputs_desc = config.model.get_inputs_desc()
            self.tower_func = TowerFuncWrapper(
                lambda *inputs: config.model.build_graph(inputs),
                self.inputs_desc)
            self._main_tower_vs_name = ""

            def gp(input_names, output_names, tower=0):
                return TowerTrainer.get_predictor(self, input_names, output_names, device=tower)
            self.get_predictor = gp

            old_train = self.train

            def train():
                return old_train(
                    config.callbacks, config.monitors,
                    config.session_creator, config.session_init,
                    config.steps_per_epoch, config.starting_epoch, config.max_epoch)
            self.train = train

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

    @call_only_once
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

    @call_only_once
    def initialize(self, session_creator, session_init):
        """
        Initialize self.sess and self.hooked_sess.
        Must be called after callbacks are setup.
        """
        session_init._setup_graph()

        logger.info("Creating the session ...")

        hooks = self._callbacks.get_hooks()
        self.sess = session_creator.create_session()
        self.hooked_sess = tf.train.MonitoredSession(
            session_creator=ReuseSessionCreator(self.sess), hooks=hooks)

        if self.is_chief:
            logger.info("Initializing the session ...")
            session_init._run_init(self.sess)
        else:
            if not isinstance(self._config.session_init, JustCurrentSession):
                logger.warn("This is not a chief worker, 'session_init' was ignored!")

        self.sess.graph.finalize()
        logger.info("Graph Finalized.")

    @call_only_once
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

    # create the old trainer when called with TrainConfig
    def __new__(cls, *args, **kwargs):
        if (len(args) > 0 and isinstance(args[0], old_train.TrainConfig)) \
                or 'config' in kwargs:
            name = cls.__name__
            try:
                old_trainer = getattr(old_train, name)
            except AttributeError:
                # custom trainer. has to live with it
                return super(Trainer, cls).__new__(cls)
            else:
                logger.warn("You're creating trainers with old trainer API!")
                logger.warn("Now it returns the old trainer for you, please switch to the new API!")
                logger.warn("'SomeTrainer(config, ...).train()' should be equivalent to "
                            "'launch_train_with_config(config, SomeTrainer(...))' in the new API.")
                return old_trainer(*args, **kwargs)
        else:
            return super(Trainer, cls).__new__(cls)


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

    To use a SingleCostTrainer object, call `trainer.setup_graph(...); trainer.train(...)`.
    """

    def train(self,
              callbacks, monitors,
              session_creator, session_init,
              steps_per_epoch, starting_epoch, max_epoch):
        """
        Same as :meth:`Trainer.train()`, except that the callbacks this
        trainer needs are automatically added.
        """
        callbacks = callbacks + self._internal_callbacks
        super(SingleCostTrainer, self).train(
            callbacks, monitors,
            session_creator, session_init,
            steps_per_epoch, starting_epoch, max_epoch)

    @call_only_once
    def setup_graph(self, inputs_desc, input, get_cost_fn, get_opt_fn):
        """
        Responsible for building the main training graph for single-cost training.

        Args:
            inputs_desc ([InputDesc]):
            input (InputSource):
            get_cost_fn ([tf.Tensor] -> tf.Tensor): callable, takes some input tenosrs and return a cost tensor.
                Might get called multiple times for data-parallel training or inference.
            get_opt_fn (-> tf.train.Optimizer): callable which returns an
                optimizer. Will only be called once.

        Returns:
            [Callback]: a (possibly empty) list of callbacks needed for training.
                These callbacks will be automatically added when you call `train()`.
                So you can usually ignore the return value.
        """
        get_cost_fn = TowerFuncWrapper(get_cost_fn, inputs_desc)
        get_opt_fn = memoized(get_opt_fn)
        self.set_tower_func(get_cost_fn)

        input_callbacks = self._setup_input(inputs_desc, input)
        train_callbacks = self._setup_graph(input, get_cost_fn, get_opt_fn)
        self._internal_callbacks = input_callbacks + train_callbacks
        return self._internal_callbacks

    @abstractmethod
    def _setup_graph(self, input, get_cost_fn, get_opt_fn):
        pass

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
        assert isinstance(input, FeedfreeInput), input

        def get_grad_fn():
            ctx = get_current_tower_context()
            cost = get_cost_fn(*input.get_input_tensors())

            varlist = ctx.filter_vars_by_vs_name(tf.trainable_variables())
            opt = get_opt_fn()
            grads = opt.compute_gradients(
                cost, var_list=varlist,
                gate_gradients=False, colocate_gradients_with_ops=True)
            grads = FilterNoneGrad().process(grads)
            return grads

        return get_grad_fn
