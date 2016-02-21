#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# File: config.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import tensorflow as tf

from ..callbacks import Callbacks
from ..models import ModelDesc
from ..utils import *
from ..dataflow import DataFlow

__all__ = ['TrainConfig']

class TrainConfig(object):
    """
    Config for training a model with a single loss
    """
    def __init__(self, **kwargs):
        """
        Args:
            dataset: the dataset to train. a tensorpack.dataflow.DataFlow instance.
            optimizer: a tf.train.Optimizer instance defining the optimizer for trainig.
            callbacks: a tensorpack.utils.callback.Callbacks instance. Define
                the callbacks to perform during training. has to contain a
                SummaryWriter and a PeriodicSaver
            session_config: a tf.ConfigProto instance to instantiate the
                session. default to a session running 1 GPU.
            session_init: a tensorpack.utils.sessinit.SessionInit instance to
                initialize variables of a session. default to a new session.
            model: a ModelDesc instance
            step_per_epoch: the number of steps (parameter updates) to perform
                in each epoch. default to dataset.size()
            max_epoch: maximum number of epoch to run training. default to 100
            nr_tower: int. number of towers. default to 1.
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
        self.session_init = kwargs.pop('session_init', NewSession())
        assert_type(self.session_init, SessionInit)
        self.step_per_epoch = int(kwargs.pop('step_per_epoch', self.dataset.size()))
        self.max_epoch = int(kwargs.pop('max_epoch', 100))
        assert self.step_per_epoch > 0 and self.max_epoch > 0
        self.nr_tower = int(kwargs.pop('nr_tower', 1))
        assert len(kwargs) == 0, 'Unknown arguments: {}'.format(str(kwargs.keys()))

