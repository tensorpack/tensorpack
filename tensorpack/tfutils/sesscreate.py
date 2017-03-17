#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: sesscreate.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import tensorflow as tf
from .common import get_default_sess_config

__all__ = ['NewSessionCreator', 'ReuseSessionCreator', 'SessionCreatorAdapter']


class NewSessionCreator(tf.train.SessionCreator):
    def __init__(self, target='', graph=None, config=None):
        """
        Args:
            target, graph, config: same as :meth:`Session.__init__()`.
            config: defaults to :func:`tfutils.get_default_sess_config()`
        """
        self.target = target
        if config is None:
            config = get_default_sess_config()
        self.config = config
        self.graph = graph

    def create_session(self):
        return tf.Session(target=self.target, graph=self.graph, config=self.config)


class ReuseSessionCreator(tf.train.SessionCreator):
    def __init__(self, sess):
        """
        Args:
            sess (tf.Session): the session to reuse
        """
        self.sess = sess

    def create_session(self):
        return self.sess


class SessionCreatorAdapter(tf.train.SessionCreator):
    def __init__(self, session_creator, func):
        """
        Args:
            session_creator (tf.train.SessionCreator): a session creator
            func (tf.Session -> tf.Session): takes a session created by
            ``session_creator``, and return a new session to be returned by ``self.create_session``
        """
        self._creator = session_creator
        self._func = func

    def create_session(self):
        sess = self._creator.create_session()
        return self._func(sess)
