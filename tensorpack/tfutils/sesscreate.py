# -*- coding: utf-8 -*-
# File: sesscreate.py


import tensorflow as tf
from .common import get_default_sess_config
from ..utils import logger

__all__ = ['NewSessionCreator', 'ReuseSessionCreator', 'SessionCreatorAdapter']

"""
A SessionCreator should:
    create the session
    initialize all variables
    return a session that is ready to use
    not finalize the graph
"""


class NewSessionCreator(tf.train.SessionCreator):
    def __init__(self, target='', config=None):
        """
        Args:
            target, config: same as :meth:`Session.__init__()`.
            config: a :class:`tf.ConfigProto` instance, defaults to :func:`tfutils.get_default_sess_config()`
        """
        self.target = target

        if config is None:
            # distributed trainer doesn't support user-provided config
            # we set this attribute so that they can check
            self.user_provided_config = False
            config = get_default_sess_config()
        else:
            self.user_provided_config = True
            logger.warn(
                "User-provided custom session config may not work due to TF \
bugs. See https://github.com/tensorpack/tensorpack/issues/497 for workarounds.")
        self.config = config

    def create_session(self):
        sess = tf.Session(target=self.target, config=self.config)
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        return sess


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
