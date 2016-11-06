# -*- coding: utf-8 -*-
# File: config.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import tensorflow as tf

from ..callbacks import Callbacks
from ..models import ModelDesc
from ..utils import logger
from ..tfutils import (JustCurrentSession,
        get_default_sess_config, SessionInit)
from ..dataflow import DataFlow

__all__ = ['TrainConfig']

class TrainConfig(object):
    """
    Config for training a model with a single loss
    """
    def __init__(self, **kwargs):
        """
        :param dataset: the dataset to train. a `DataFlow` instance.
        :param optimizer: a `tf.train.Optimizer` instance defining the optimizer for trainig.
        :param callbacks: a `callback.Callbacks` instance. Define
            the callbacks to perform during training.
        :param session_config: a `tf.ConfigProto` instance to instantiate the session.
        :param session_init: a `sessinit.SessionInit` instance to
            initialize variables of a session. default to a new session.
        :param model: a `ModelDesc` instance.
        :param starting_epoch: int. default to be 1.
        :param step_per_epoch: the number of steps (SGD updates) to perform in each epoch.
        :param max_epoch: maximum number of epoch to run training. default to inf
        :param nr_tower: int. number of training towers. default to 1.
        :param tower: list of training towers in relative id. default to `range(nr_tower)` if nr_tower is given.
        """
        def assert_type(v, tp):
            assert isinstance(v, tp), v.__class__
        self.dataset = kwargs.pop('dataset')
        assert_type(self.dataset, DataFlow)
        self.optimizer = kwargs.pop('optimizer')
        assert_type(self.optimizer, tf.train.Optimizer)
        self.callbacks = kwargs.pop('callbacks')
        assert_type(self.callbacks, Callbacks)
        self.model = kwargs.pop('model')
        assert_type(self.model, ModelDesc)

        self.session_config = kwargs.pop('session_config', get_default_sess_config())
        assert_type(self.session_config, tf.ConfigProto)
        self.session_init = kwargs.pop('session_init', JustCurrentSession())
        assert_type(self.session_init, SessionInit)

        self.step_per_epoch = kwargs.pop('step_per_epoch', None)
        if self.step_per_epoch is None:
            try:
                self.step_per_epoch = self.dataset.size()
            except NotImplementedError:
                logger.exception("You must set `step_per_epoch` if dataset.size() is not implemented.")
        else:
            self.step_per_epoch = int(self.step_per_epoch)

        self.starting_epoch = int(kwargs.pop('starting_epoch', 1))
        self.max_epoch = int(kwargs.pop('max_epoch', 99999))
        assert self.step_per_epoch >= 0 and self.max_epoch > 0

        if 'nr_tower' in kwargs:
            assert 'tower' not in kwargs, "Cannot set both nr_tower and tower in TrainConfig!"
            self.nr_tower = kwargs.pop('nr_tower')
        elif 'tower' in kwargs:
            self.tower = kwargs.pop('tower')
        else:
            self.tower = [0]

        self.extra_threads_procs = kwargs.pop('extra_threads_procs', [])
        if self.extra_threads_procs:
            logger.warn("[DEPRECATED] use the Callback StartProcOrThread instead of _extra_threads_procs")
            from ..callbacks.concurrency import StartProcOrThread
            self.callbacks.append(StartProcOrThread(self.extra_threads_procs))
        assert len(kwargs) == 0, 'Unknown arguments: {}'.format(str(kwargs.keys()))

    def set_tower(self, nr_tower=None, tower=None):
        # this is a deprecated function
        logger.warn("config.set_tower is deprecated. set config.tower or config.nr_tower directly")
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
