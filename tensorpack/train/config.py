# -*- coding: utf-8 -*-
# File: config.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

from ..callbacks import (
    MovingAverageSummary,
    ProgressBar, MergeAllSummaries,
    TFEventWriter, JSONWriter, ScalarPrinter, RunUpdateOps)
from ..dataflow.base import DataFlow
from ..graph_builder.model_desc import ModelDescBase
from ..utils import logger
from ..utils.develop import log_deprecated
from ..tfutils import (JustCurrentSession,
                       get_default_sess_config, SessionInit)
from ..tfutils.sesscreate import NewSessionCreator
from ..graph_builder.input_source_base import InputSource

__all__ = ['TrainConfig']


class TrainConfig(object):
    """
    Config for trainer.
    """

    def __init__(self,
                 dataflow=None, data=None, model=None,
                 callbacks=None, extra_callbacks=None, monitors=None,
                 session_creator=None, session_config=None, session_init=None,
                 starting_epoch=1, steps_per_epoch=None, max_epoch=99999,
                 nr_tower=1, tower=None, predict_tower=None,
                 **kwargs):
        """
        Note:
            It depends on the specific trainer what fields are necessary.
            Most existing trainers in tensorpack requires one of `dataflow` or `data`,
            and `model` to be present in the config.

        Args:
            dataflow (DataFlow):
            data (InputSource):
            model (ModelDescBase):

            callbacks (list): a list of :class:`Callback` to perform during training.
            extra_callbacks (list): the same as ``callbacks``. This argument
                is only used to provide the defaults in addition to ``callbacks``. The defaults are
                ``MovingAverageSummary()``, ``ProgressBar()``,
                ``MergeAllSummaries()``, ``RunUpdateOps()``. The list of
                callbacks that will be used in the end is ``callbacks + extra_callbacks``.
            monitors (list): a list of :class:`TrainingMonitor`.
                Defaults to ``TFEventWriter()``, ``JSONWriter()``, ``ScalarPrinter()``.

            session_creator (tf.train.SessionCreator): Defaults to :class:`sesscreate.NewSessionCreator()`
                with the config returned by :func:`tfutils.get_default_sess_config()`.
            session_config (tf.ConfigProto): when session_creator is None, use this to create the session.
            session_init (SessionInit): how to initialize variables of a session. Defaults to do nothing.

            starting_epoch (int): The index of the first epoch.
            steps_per_epoch (int): the number of steps (defined by :meth:`Trainer.run_step`) to run in each epoch.
                Defaults to the input data size.
            max_epoch (int): maximum number of epoch to run training.

            nr_tower (int): number of training towers, used by multigpu trainers.
            tower ([int]): list of training towers in relative GPU id.
        """

        # TODO type checker decorator
        def assert_type(v, tp):
            assert isinstance(v, tp), v.__class__

        # process data & model
        if 'dataset' in kwargs:
            dataflow = kwargs.pop('dataset')
            log_deprecated("TrainConfig.dataset", "Use TrainConfig.dataflow instead.", "2017-09-11")
        assert data is None or dataflow is None, "dataflow and data cannot be both presented in TrainConfig!"
        if dataflow is not None:
            assert_type(dataflow, DataFlow)
        if data is not None:
            assert_type(data, InputSource)
        self.dataflow = dataflow
        self.data = data

        if model is not None:
            assert_type(model, ModelDescBase)
        self.model = model

        if callbacks is None:
            callbacks = []
        assert_type(callbacks, list)
        if extra_callbacks is None:
            extra_callbacks = [
                MovingAverageSummary(),
                ProgressBar(),
                MergeAllSummaries(),
                RunUpdateOps()]
        self._callbacks = callbacks + extra_callbacks

        if monitors is None:
            monitors = [TFEventWriter(), JSONWriter(), ScalarPrinter()]
        self.monitors = monitors

        if session_init is None:
            session_init = JustCurrentSession()
        self.session_init = session_init
        assert_type(self.session_init, SessionInit)

        if session_creator is None:
            if session_config is not None:
                self.session_creator = NewSessionCreator(config=session_config)
            else:
                self.session_creator = NewSessionCreator(config=get_default_sess_config())
        else:
            self.session_creator = session_creator
            assert session_config is None, "Cannot set both session_creator and session_config!"
        self.session_config = session_config

        if steps_per_epoch is None:
            try:
                if dataflow is not None:
                    steps_per_epoch = dataflow.size()
                elif data is not None:
                    steps_per_epoch = data.size()
                else:
                    raise NotImplementedError()
            except NotImplementedError:
                logger.error("You must set `TrainConfig(steps_per_epoch)` if data.size() is not available.")
                raise
        else:
            steps_per_epoch = int(steps_per_epoch)
        self.steps_per_epoch = steps_per_epoch

        self.starting_epoch = int(starting_epoch)
        self.max_epoch = int(max_epoch)
        assert self.steps_per_epoch > 0 and self.max_epoch > 0

        nr_tower = max(nr_tower, 1)
        self.nr_tower = nr_tower
        if tower is not None:
            assert self.nr_tower == 1, "Cannot set both nr_tower and tower in TrainConfig!"
            self.tower = tower

        if predict_tower is None:
            predict_tower = [0]
        self.predict_tower = predict_tower
        if isinstance(self.predict_tower, int):
            self.predict_tower = [self.predict_tower]

        assert len(kwargs) == 0, 'Unknown arguments: {}'.format(str(kwargs.keys()))

    @property
    def nr_tower(self):
        return len(self.tower)

    @nr_tower.setter
    def nr_tower(self, value):
        self.tower = list(range(value))

    @property
    def callbacks(self):        # disable setter
        return self._callbacks
