# -*- coding: utf-8 -*-
# File: config.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import tensorflow as tf

from ..callbacks import (
    Callbacks, MovingAverageSummary,
    ProgressBar, MergeAllSummaries)
from ..dataflow.base import DataFlow
from ..models import ModelDesc
from ..utils import logger
from ..utils.develop import log_deprecated
from ..tfutils import (JustCurrentSession,
                       get_default_sess_config, SessionInit)
from ..tfutils.optimizer import apply_grad_processors
from .input_data import InputData
from .monitor import TFSummaryWriter, JSONWriter, ScalarPrinter

__all__ = ['TrainConfig']


class TrainConfig(object):
    """
    Config for trainer.
    """

    def __init__(self,
                 dataflow=None, data=None,
                 model=None,
                 callbacks=None, extra_callbacks=None,
                 monitors=None,
                 session_config=get_default_sess_config(), session_init=None,
                 starting_epoch=1, steps_per_epoch=None, max_epoch=99999,
                 nr_tower=1, tower=None, predict_tower=[0],
                 **kwargs):
        """
        Args:
            dataflow (DataFlow): the dataflow to train.
            data (InputData): an `InputData` instance. Only one of ``dataflow``
                or ``data`` has to be present.
            model (ModelDesc): the model to train.
            callbacks (list): a list of :class:`Callback` to perform during training.
            extra_callbacks (list): the same as ``callbacks``. This argument
                is only used to provide the defaults. The defaults are
                ``[MovingAverageSummary(), ProgressBar(), MergeAllSummaries()]``. The list of
                callbacks that will be used in the end are ``callbacks + extra_callbacks``.
            monitors (list): a list of :class:`TrainingMonitor`.
                Defaults to ``[TFSummaryWriter(), JSONWriter(), ScalarPrinter()]``.
            session_config (tf.ConfigProto): the config used to instantiate the session.
            session_init (SessionInit): how to initialize variables of a session. Defaults to a new session.
            starting_epoch (int): The index of the first epoch.
            steps_per_epoch (int): the number of steps (defined by :meth:`Trainer.run_step`) to run in each epoch.
                Defaults to the input data size.
            max_epoch (int): maximum number of epoch to run training.
            nr_tower (int): number of training towers.
            tower (list of int): list of training towers in relative id.
            predict_tower (list of int): list of prediction towers in their relative gpu id. Use -1 for cpu.
        """

        # TODO type checker decorator
        def assert_type(v, tp):
            assert isinstance(v, tp), v.__class__

        # process data
        if 'dataset' in kwargs:
            dataflow = kwargs.pop('dataset')
            log_deprecated("TrainConfig.dataset", "Use TrainConfig.dataflow instead.")
        if dataflow is not None:
            assert data is None, "dataflow and data cannot be both presented in TrainConfig!"
            self.dataflow = dataflow
            assert_type(self.dataflow, DataFlow)
            self.data = None
        else:
            self.data = data
            assert_type(self.data, InputData)
            self.dataflow = None

        if isinstance(callbacks, Callbacks):
            # keep quiet now because I haven't determined the final API yet.
            log_deprecated(
                "TrainConfig(callbacks=Callbacks([...]))",
                "Change the argument 'callbacks=' to a *list* of callbacks without StatPrinter().")

            callbacks = callbacks.cbs[:-1]  # the last one is StatPrinter()
        assert_type(callbacks, list)
        if extra_callbacks is None:
            extra_callbacks = [
                MovingAverageSummary(),
                ProgressBar(),
                MergeAllSummaries()]
        self.callbacks = callbacks + extra_callbacks
        assert_type(self.callbacks, list)

        if monitors is None:
            monitors = [TFSummaryWriter(), JSONWriter(), ScalarPrinter()]
        self.monitors = monitors

        self.model = model
        assert_type(self.model, ModelDesc)

        self.session_config = session_config
        assert_type(self.session_config, tf.ConfigProto)
        if session_init is None:
            session_init = JustCurrentSession()
        self.session_init = session_init
        assert_type(self.session_init, SessionInit)

        if steps_per_epoch is None:
            steps_per_epoch = kwargs.pop('step_per_epoch', None)
            if steps_per_epoch is not None:
                log_deprecated("step_per_epoch", "Use steps_per_epoch instead!", "2017-03-27")
        if steps_per_epoch is None:
            try:
                if dataflow is not None:
                    steps_per_epoch = self.dataflow.size()
                else:
                    steps_per_epoch = self.data.size()
            except NotImplementedError:
                logger.exception("You must set `steps_per_epoch` if dataset.size() is not implemented.")
        else:
            steps_per_epoch = int(steps_per_epoch)
        self.steps_per_epoch = steps_per_epoch

        self.starting_epoch = int(starting_epoch)
        self.max_epoch = int(max_epoch)
        assert self.steps_per_epoch >= 0 and self.max_epoch > 0

        self.nr_tower = nr_tower
        if tower is not None:
            assert self.nr_tower == 1, "Cannot set both nr_tower and tower in TrainConfig!"
            self.tower = tower

        self.predict_tower = predict_tower
        if isinstance(self.predict_tower, int):
            self.predict_tower = [self.predict_tower]
        assert len(set(self.predict_tower)) == len(self.predict_tower), \
            "Cannot have duplicated predict_tower!"

        if 'optimizer' in kwargs:
            log_deprecated("TrainConfig(optimizer=...)",
                           "Use ModelDesc._get_optimizer() instead.",
                           "2017-04-12")
            self._optimizer = kwargs.pop('optimizer')
            assert_type(self._optimizer, tf.train.Optimizer)
        else:
            self._optimizer = None

        assert len(kwargs) == 0, 'Unknown arguments: {}'.format(str(kwargs.keys()))

    def set_tower(self, nr_tower=None, tower=None):
        log_deprecated("config.set_tower", "Set config.tower or config.nr_tower directly.", "2017-03-15")
        assert nr_tower is None or tower is None, "Cannot set both nr_tower and tower!"
        if nr_tower:
            tower = list(range(nr_tower))
        else:
            if isinstance(tower, int):
                tower = list(range(tower))
        self.tower = tower
        assert isinstance(self.tower, list)

    @property
    def nr_tower(self):
        return len(self.tower)

    @nr_tower.setter
    def nr_tower(self, value):
        self.tower = list(range(value))

    @property
    def optimizer(self):
        """ for back-compatibilty only. will remove in the future"""
        if self._optimizer:
            opt = self._optimizer
        else:
            opt = self.model.get_optimizer()
        gradproc = self.model.get_gradient_processor()
        if gradproc:
            log_deprecated("ModelDesc.get_gradient_processor()",
                           "Use gradient processor to build an optimizer instead.", "2017-04-12")
            opt = apply_grad_processors(opt, gradproc)
        if not self._optimizer:
            self._optimizer = opt
        return opt
