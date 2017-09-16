# -*- coding: UTF-8 -*-
# File: sessinit.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import os
import numpy as np
import tensorflow as tf
import six

from ..utils import logger
from .common import get_op_tensor_name
from .varmanip import (SessionUpdate, get_savename_from_varname,
                       is_training_name, get_checkpoint_path)

__all__ = ['SessionInit', 'ChainInit',
           'SaverRestore', 'SaverRestoreRelaxed', 'DictRestore',
           'JustCurrentSession', 'get_model_loader', 'TryResumeTraining']


class SessionInit(object):
    """ Base class for utilities to initialize a (existing) session. """
    def init(self, sess):
        """
        Initialize a session

        Args:
            sess (tf.Session): the session
        """
        self._init(sess)

    def _init(self, sess):
        self._setup_graph()
        self._run_init(sess)

    def _setup_graph(self):
        pass

    def _run_init(self, sess):
        pass


class JustCurrentSession(SessionInit):
    """ This is a no-op placeholder"""
    pass


class CheckpointReaderAdapter(object):
    """
    An adapter to work around old checkpoint format, where the keys are op
    names instead of tensor names (with :0).
    """
    def __init__(self, reader):
        self._reader = reader
        m = self._reader.get_variable_to_shape_map()
        self._map = {k if k.endswith(':0') else k + ':0': v
                     for k, v in six.iteritems(m)}

    def get_variable_to_shape_map(self):
        return self._map

    def get_tensor(self, name):
        if self._reader.has_tensor(name):
            return self._reader.get_tensor(name)
        if name in self._map:
            assert name.endswith(':0'), name
            name = name[:-2]
        return self._reader.get_tensor(name)

    def has_tensor(self, name):
        return name in self._map

    # some checkpoint might not have ':0'
    def get_real_name(self, name):
        if self._reader.has_tensor(name):
            return name
        assert self.has_tensor(name)
        return name[:-2]


class MismatchLogger(object):
    def __init__(self, exists, nonexists):
        self._exists = exists
        self._nonexists = nonexists
        self._names = []

    def add(self, name):
        self._names.append(name)

    def log(self):
        if len(self._names):
            logger.warn("The following variables are in the {}, but not found in the {}: {}".format(
                self._exists, self._nonexists, ', '.join(self._names)))


class SaverRestore(SessionInit):
    """
    Restore a tensorflow checkpoint saved by :class:`tf.train.Saver` or :class:`ModelSaver`.
    """
    def __init__(self, model_path, prefix=None, ignore=[]):
        """
        Args:
            model_path (str): a model name (model-xxxx) or a ``checkpoint`` file.
            prefix (str): during restore, add a ``prefix/`` for every variable in this checkpoint
            ignore (list[str]): list of tensor names that should be ignored during loading, e.g. learning-rate
        """
        if model_path.endswith('.npy') or model_path.endswith('.npz'):
            logger.warn("SaverRestore expect a TF checkpoint, but got a model path '{}'.".format(model_path) +
                        " To load from a dict, use 'DictRestore'.")
        model_path = get_checkpoint_path(model_path)
        self.path = model_path
        self.prefix = prefix
        self.ignore = [i if i.endswith(':0') else i + ':0' for i in ignore]

    def _setup_graph(self):
        dic = self._get_restore_dict()
        self.saver = tf.train.Saver(var_list=dic, name=str(id(dic)))

    def _run_init(self, sess):
        logger.info("Restoring checkpoint from {} ...".format(self.path))
        self.saver.restore(sess, self.path)

    @staticmethod
    def _read_checkpoint_vars(model_path):
        """ return a set of strings """
        reader = tf.train.NewCheckpointReader(model_path)
        reader = CheckpointReaderAdapter(reader)    # use an adapter to standardize the name
        ckpt_vars = reader.get_variable_to_shape_map().keys()
        return reader, set(ckpt_vars)

    def _match_vars(self, func):
        reader, chkpt_vars = SaverRestore._read_checkpoint_vars(self.path)
        graph_vars = tf.global_variables()
        chkpt_vars_used = set()

        mismatch = MismatchLogger('graph', 'checkpoint')
        for v in graph_vars:
            name = get_savename_from_varname(v.name, varname_prefix=self.prefix)
            if name in self.ignore and reader.has_tensor(name):
                logger.info("Variable {} in the graph will be not loaded from the checkpoint!".format(name))
            else:
                if reader.has_tensor(name):
                    func(reader, name, v)
                    chkpt_vars_used.add(name)
                else:
                    vname = v.op.name
                    if not is_training_name(vname):
                        mismatch.add(vname)
        mismatch.log()
        mismatch = MismatchLogger('checkpoint', 'graph')
        if len(chkpt_vars_used) < len(chkpt_vars):
            unused = chkpt_vars - chkpt_vars_used
            for name in sorted(unused):
                if not is_training_name(name):
                    mismatch.add(name)
        mismatch.log()

    def _get_restore_dict(self):
        var_dict = {}

        def f(reader, name, v):
            name = reader.get_real_name(name)
            assert name not in var_dict, "Restore conflict: {} and {}".format(v.name, var_dict[name].name)
            var_dict[name] = v
        self._match_vars(f)
        return var_dict


class SaverRestoreRelaxed(SaverRestore):
    """ Same as :class:`SaverRestore`, but has more relaxed constraints.

        It allows upcasting certain variables, or reshape certain
        variables when there is a mismatch that can be fixed.
        Another advantage is that it doesn't add any new ops to the graph.
        But it is also slower than :class:`SaverRestore`.
    """
    def _run_init(self, sess):
        logger.info(
            "Restoring checkpoint from {} ...".format(self.path))

        def f(reader, name, v):
            val = reader.get_tensor(name)
            SessionUpdate.load_value_to_var(v, val)
        with sess.as_default():
            self._match_vars(f)


class DictRestore(SessionInit):
    """
    Restore variables from a dictionary.
    """

    def __init__(self, variable_dict):
        """
        Args:
            variable_dict (dict): a dict of {name: value}
        """
        assert isinstance(variable_dict, dict), type(variable_dict)
        # use varname (with :0) for consistency
        self._prms = {get_op_tensor_name(n)[1]: v for n, v in six.iteritems(variable_dict)}

    def _run_init(self, sess):
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

        variable_names = set([k.name for k in variables])
        param_names = set(six.iterkeys(self._prms))

        intersect = variable_names & param_names

        logger.info("Variables to restore from dict: {}".format(', '.join(map(str, intersect))))

        mismatch = MismatchLogger('graph', 'dict')
        for k in sorted(variable_names - param_names):
            if not is_training_name(k):
                mismatch.add(k)
        mismatch.log()
        mismatch = MismatchLogger('dict', 'graph')
        for k in sorted(param_names - variable_names):
            mismatch.add(k)
        mismatch.log()

        upd = SessionUpdate(sess, [v for v in variables if v.name in intersect])
        logger.info("Restoring from dict ...")
        upd.update({name: value for name, value in six.iteritems(self._prms) if name in intersect})


class ChainInit(SessionInit):
    """ Initialize a session by a list of :class:`SessionInit` instance, executed one by one.
    This can be useful for, e.g., loading several models from different files
    to form a composition of models.
    """

    def __init__(self, sess_inits):
        """
        Args:
            sess_inits (list[SessionInit]): list of :class:`SessionInit` instances.
        """
        self.inits = sess_inits

    def _init(self, sess):
        for i in self.inits:
            i.init(sess)

    def _setup_graph(self):
        for i in self.inits:
            i._setup_graph()

    def _run_init(self, sess):
        for i in self.inits:
            i._run_init(sess)


def get_model_loader(filename):
    """
    Get a corresponding model loader by looking at the file name.

    Returns:
        SessInit: either a :class:`DictRestore` (if name ends with 'npy/npz') or
        :class:`SaverRestore` (otherwise).
    """
    if filename.endswith('.npy'):
        assert tf.gfile.Exists(filename), filename
        return DictRestore(np.load(filename, encoding='latin1').item())
    elif filename.endswith('.npz'):
        assert tf.gfile.Exists(filename), filename
        obj = np.load(filename)
        return DictRestore(dict(obj))
    else:
        return SaverRestore(filename)


def TryResumeTraining():
    """
    Try loading latest checkpoint from ``logger.LOG_DIR``, only if there is one.

    Returns:
        SessInit: either a :class:`JustCurrentSession`, or a :class:`SaverRestore`.
    """
    if not logger.LOG_DIR:
        return JustCurrentSession()
    path = os.path.join(logger.LOG_DIR, 'checkpoint')
    if not tf.gfile.Exists(path):
        return JustCurrentSession()
    return SaverRestore(path)
