# -*- coding: UTF-8 -*-
# File: sessinit.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import os
from abc import abstractmethod, ABCMeta
import numpy as np
import tensorflow as tf
import six

from ..utils import logger, PREDICT_TOWER
from .common import get_op_tensor_name
from .varmanip import (SessionUpdate, get_savename_from_varname,
                       is_training_name, get_checkpoint_path)

__all__ = ['SessionInit', 'NewSession', 'SaverRestore',
           'ParamRestore', 'ChainInit',
           'JustCurrentSession', 'get_model_loader']

# TODO they initialize_all at the beginning by default.


@six.add_metaclass(ABCMeta)
class SessionInit(object):
    """ Base class for utilities to initialize a session. """

    def init(self, sess):
        """
        Initialize a session

        Args:
            sess (tf.Session): the session
        """
        self._init(sess)

    @abstractmethod
    def _init(self, sess):
        pass


class JustCurrentSession(SessionInit):
    """ This is a no-op placeholder"""

    def _init(self, sess):
        pass


class NewSession(SessionInit):
    """
    Initialize global variables by their initializer.
    """

    def _init(self, sess):
        sess.run(tf.global_variables_initializer())


class SaverRestore(SessionInit):
    """
    Restore a tensorflow checkpoint saved by :class:`tf.train.Saver` or :class:`ModelSaver`.
    """

    def __init__(self, model_path, prefix=None):
        """
        Args:
            model_path (str): a model name (model-xxxx) or a ``checkpoint`` file.
            prefix (str): during restore, add a ``prefix/`` for every variable in this checkpoint
        """
        model_path = get_checkpoint_path(model_path)
        self.path = model_path
        self.prefix = prefix

    def _init(self, sess):
        logger.info(
            "Restoring checkpoint from {} ...".format(self.path))
        reader, chkpt_vars = SaverRestore._read_checkpoint_vars(self.path)
        graph_vars = tf.global_variables()
        chkpt_vars_used = set()

        with sess.as_default():
            for v in graph_vars:
                name = get_savename_from_varname(v.name, varname_prefix=self.prefix)
                if name in chkpt_vars:
                    val = reader.get_tensor(name)
                    SessionUpdate.load_value_to_var(v, val)
                    chkpt_vars_used.add(name)
                else:
                    vname = v.op.name
                    if not is_training_name(vname):
                        logger.warn("Variable {} in the graph not found in checkpoint!".format(vname))
            if len(chkpt_vars_used) < len(chkpt_vars):
                unused = chkpt_vars - chkpt_vars_used
                for name in sorted(unused):
                    if not is_training_name(name):
                        logger.warn("Variable {} in checkpoint not found in the graph!".format(name))

    @staticmethod
    def _read_checkpoint_vars(model_path):
        """ return a set of strings """
        reader = tf.train.NewCheckpointReader(model_path)
        ckpt_vars = reader.get_variable_to_shape_map().keys()
        for v in ckpt_vars:
            if v.startswith(PREDICT_TOWER):
                logger.error("Found {} in checkpoint. "
                             "But anything from prediction tower shouldn't be saved.".format(v.name))
        return reader, set(ckpt_vars)


class ParamRestore(SessionInit):
    """
    Restore variables from a dictionary.
    """

    def __init__(self, param_dict):
        """
        Args:
            param_dict (dict): a dict of {name: value}
        """
        # use varname (with :0) for consistency
        self.prms = {get_op_tensor_name(n)[1]: v for n, v in six.iteritems(param_dict)}

    def _init(self, sess):
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)  # TODO

        variable_names = set([get_savename_from_varname(k.name) for k in variables])
        param_names = set(six.iterkeys(self.prms))

        intersect = variable_names & param_names

        logger.info("Params to restore: {}".format(
            ', '.join(map(str, intersect))))
        for k in sorted(variable_names - param_names):
            if not is_training_name(k):
                logger.warn("Variable {} in the graph not found in the dict!".format(k))
        for k in sorted(param_names - variable_names):
            logger.warn("Variable {} in the dict not found in the graph!".format(k))

        upd = SessionUpdate(sess,
                            [v for v in variables if
                             get_savename_from_varname(v.name) in intersect])
        logger.info("Restoring from dict ...")
        upd.update({name: value for name, value in six.iteritems(self.prms) if name in intersect})


class ChainInit(SessionInit):
    """ Initialize a session by a list of :class:`SessionInit` instance, executed one by one.
    This can be useful for, e.g., loading several models from different files
    to form a composition of models.
    """

    def __init__(self, sess_inits, new_session=True):
        """
        Args:
            sess_inits (list): list of :class:`SessionInit` instances.
            new_session (bool): add a ``NewSession()`` and the beginning, if
                not there.
        """
        if new_session and not isinstance(sess_inits[0], NewSession):
            sess_inits.insert(0, NewSession())
        self.inits = sess_inits

    def _init(self, sess):
        for i in self.inits:
            i.init(sess)


def get_model_loader(filename):
    """
    Get a corresponding model loader by looking at the file name.

    Returns:
        SessInit: either a :class:`ParamRestore` (if name ends with 'npy') or
        :class:`SaverRestore` (otherwise).
    """
    if filename.endswith('.npy'):
        assert os.path.isfile(filename), filename
        return ParamRestore(np.load(filename, encoding='latin1').item())
    else:
        return SaverRestore(filename)
