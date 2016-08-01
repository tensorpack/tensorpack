# -*- coding: UTF-8 -*-
# File: sessinit.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import os
from abc import abstractmethod, ABCMeta
import numpy as np
from collections import defaultdict
import re
import tensorflow as tf
import six

from ..utils import logger, EXTRA_SAVE_VARS_KEY
from .common import get_op_var_name
from .sessupdate import SessionUpdate

__all__ = ['SessionInit', 'NewSession', 'SaverRestore',
           'ParamRestore', 'ChainInit',
           'JustCurrentSession',
           'dump_session_params']

# TODO they initialize_all at the beginning by default.

class SessionInit(object):
    """ Base class for utilities to initialize a session"""
    __metaclass__ = ABCMeta

    def init(self, sess):
        """ Initialize a session

        :param sess: a `tf.Session`
        """
        self._init(sess)

    @abstractmethod
    def _init(self, sess):
        pass

class JustCurrentSession(SessionInit):
    """ Just use the current default session. This is a no-op placeholder"""
    def _init(self, sess):
        pass

class NewSession(SessionInit):
    """
    Create a new session. All variables will be initialized by their
    initializer.
    """
    def _init(self, sess):
        sess.run(tf.initialize_all_variables())

class SaverRestore(SessionInit):
    """
    Restore an old model saved by `ModelSaver`.
    """
    def __init__(self, model_path, prefix=None):
        """
        :param model_path: a model file or a ``checkpoint`` file.
        :param prefix: add a `prefix/` for every variable in this checkpoint
        """
        assert os.path.isfile(model_path)
        if os.path.basename(model_path) == 'checkpoint':
            model_path = tf.train.get_checkpoint_state(
                os.path.dirname(model_path)).model_checkpoint_path
            assert os.path.isfile(model_path)
        self.set_path(model_path)
        self.prefix = prefix

    def _init(self, sess):
        logger.info(
            "Restoring checkpoint from {}.".format(self.path))
        chkpt_vars = SaverRestore._read_checkpoint_vars(self.path)
        vars_map = self._get_vars_to_restore_multimap(chkpt_vars)
        for dic in SaverRestore._produce_restore_dict(vars_map):
            # multiple saver under same name scope would cause error:
            # training/saver.py: assert restore_op.name.endswith("restore_all"), restore_op.name
            saver = tf.train.Saver(var_list=dic, name=str(id(dic)))
            saver.restore(sess, self.path)

    def set_path(self, model_path):
        self.path = model_path

    @staticmethod
    def _produce_restore_dict(vars_multimap):
        """
        Produce {var_name: var} dict that can be used by `tf.train.Saver`, from a {var_name: [vars]} dict.
        """
        while len(vars_multimap):
            ret = {}
            for k in list(vars_multimap.keys()):
                v = vars_multimap[k]
                ret[k] = v[-1]
                del v[-1]
                if not len(v):
                    del vars_multimap[k]
            yield ret

    @staticmethod
    def _read_checkpoint_vars(model_path):
        """ return a set of strings """
        reader = tf.train.NewCheckpointReader(model_path)
        ckpt_vars = reader.get_variable_to_shape_map().keys()
        for v in ckpt_vars:
            if v.startswith('towerp'):
                logger.warn("Found {} in checkpoint. Anything from prediction tower shouldn't be saved.".format(v.name))
        return set(ckpt_vars)

    def _get_vars_to_restore_multimap(self, vars_available):
        """
        Get a dict of {var_name: [var, var]} to restore
        :param vars_available: varaible names available in the checkpoint, for existence checking
        """
        vars_to_restore = tf.all_variables()
        var_dict = defaultdict(list)
        chkpt_vars_used = set()
        for v in vars_to_restore:
            name = v.op.name
            if 'towerp' in name:
                logger.warn("Variable {} in prediction tower shouldn't exist.".format(v.name))
                # don't overwrite anything in the current prediction graph
                continue
            if 'tower' in name:
                new_name = re.sub('tower[p0-9]+/', '', name)
                name = new_name
            if self.prefix and name.startswith(self.prefix):
                name = name[len(self.prefix)+1:]
            if name in vars_available:
                var_dict[name].append(v)
                chkpt_vars_used.add(name)
                #vars_available.remove(name)
            else:
                logger.warn("Variable {} not found in checkpoint!".format(v.op.name))
        if len(chkpt_vars_used) < len(vars_available):
            unused = vars_available - chkpt_vars_used
            for name in unused:
                logger.warn("Variable {} in checkpoint doesn't exist in the graph!".format(name))
        return var_dict

class ParamRestore(SessionInit):
    """
    Restore variables from a dictionary.
    """
    def __init__(self, param_dict):
        """
        :param param_dict: a dict of {name: value}
        """
        self.prms = {get_op_var_name(n)[1]: v for n, v in six.iteritems(param_dict)}

    def _init(self, sess):
        variables = tf.get_collection(tf.GraphKeys.VARIABLES)

        variable_names = set([k.name for k in variables])
        param_names = set(six.iterkeys(self.prms))

        intersect = variable_names and param_names

        logger.info("Params to restore: {}".format(
            ', '.join(map(str, intersect))))
        for k in variable_names - param_names:
            logger.warn("Variable {} in the graph not getting restored!".format(k))
        for k in param_names - variable_names:
            logger.warn("Variable {} in the dict not found in this graph!".format(k))

        upd = SessionUpdate(sess, [v for v in variables if v.name in intersect])
        logger.info("Restoring from dict ...")
        upd.update({name: value for name, value in six.iteritems(self.prms) if name in intersect})


def ChainInit(SessionInit):
    """ Init a session by a list of SessionInit instance."""
    def __init__(self, sess_inits, new_session=True):
        """
        :params sess_inits: list of `SessionInit` instances.
        :params new_session: add a `NewSession()` and the beginning, if not there
        """
        if new_session and not isinstance(sess_inits[0], NewSession):
            sess_inits.insert(0, NewSession())
        self.inits = sess_inits

    def _init(self, sess):
        for i in self.inits:
            i.init(sess)

def dump_session_params(path):
    """ Dump value of all trainable variables to a dict and save to `path` as
    npy format, loadable by ParamRestore
    """
    var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    var.extend(tf.get_collection(EXTRA_SAVE_VARS_KEY))
    result = {}
    for v in var:
        name = v.name.replace(":0", "")
        result[name] = v.eval()
    logger.info("Variables to save to {}:".format(path))
    logger.info(str(result.keys()))
    np.save(path, result)
