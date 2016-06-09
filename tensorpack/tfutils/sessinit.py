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

from ..utils import logger

__all__ = ['SessionInit', 'NewSession', 'SaverRestore',
           'ParamRestore',
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
    def __init__(self, model_path):
        """
        :param model_path: a model file or a ``checkpoint`` file.
        """
        assert os.path.isfile(model_path)
        if os.path.basename(model_path) == 'checkpoint':
            model_path = tf.train.get_checkpoint_state(
                os.path.dirname(model_path)).model_checkpoint_path
            assert os.path.isfile(model_path)
        self.set_path(model_path)

    def _init(self, sess):
        logger.info(
            "Restoring checkpoint from {}.".format(self.path))
        sess.run(tf.initialize_all_variables())
        chkpt_vars = SaverRestore._read_checkpoint_vars(self.path)
        vars_map = SaverRestore._get_vars_to_restore_multimap(chkpt_vars)
        for dic in SaverRestore._produce_restore_dict(vars_map):
            saver = tf.train.Saver(var_list=dic)
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
        reader = tf.train.NewCheckpointReader(model_path)
        return set(reader.get_variable_to_shape_map().keys())

    @staticmethod
    def _get_vars_to_restore_multimap(vars_available):
        """
        Get a dict of {var_name: [var, var]} to restore
        :param vars_available: varaibles available in the checkpoint, for existence checking
        """
        # TODO warn if some variable in checkpoint is not used
        vars_to_restore = tf.all_variables()
        var_dict = defaultdict(list)
        for v in vars_to_restore:
            name = v.op.name
            if 'towerp' in name:
                logger.warn("Anything from prediction tower shouldn't be saved.")
            if 'tower' in name:
                new_name = re.sub('tower[p0-9]+/', '', name)
                name = new_name
            if name in vars_available:
                var_dict[name].append(v)
                vars_available.remove(name)
            else:
                logger.warn("Param {} not found in checkpoint! Will not restore.".format(v.op.name))
        #for name in vars_available:
            #logger.warn("Param {} in checkpoint doesn't appear in the graph!".format(name))
        return var_dict

class ParamRestore(SessionInit):
    """
    Restore trainable variables from a dictionary.
    """
    def __init__(self, param_dict):
        """
        :param param_dict: a dict of {name: value}
        """
        self.prms = param_dict

    def _init(self, sess):
        sess.run(tf.initialize_all_variables())
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        var_dict = dict([v.name, v] for v in variables)
        for name, value in six.iteritems(self.prms):
            if not name.endswith(':0'):
                name = name + ':0'
            try:
                var = var_dict[name]
            except (ValueError, KeyError):
                logger.warn("Param {} not found in this graph".format(name))
                continue
            logger.info("Restoring param {}".format(name))
            varshape = tuple(var.get_shape().as_list())
            if varshape != value.shape:
                assert np.prod(varshape) == np.prod(value.shape)
                logger.warn("Param {} is reshaped during loading!".format(name))
                value = value.reshape(varshape)
            sess.run(var.assign(value))

def dump_session_params(path):
    """ Dump value of all trainable variables to a dict and save to `path` as
    npy format, loadable by ParamRestore
    """
    var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    result = {}
    for v in var:
        name = v.name.replace(":0", "")
        result[name] = v.eval()
    logger.info("Params to save to {}:".format(path))
    logger.info(str(result.keys()))
    np.save(path, result)
