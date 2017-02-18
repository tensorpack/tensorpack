#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: varmanip.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import six
import os
import tensorflow as tf
from collections import defaultdict
import numpy as np
from ..utils import logger
from ..utils.naming import PREDICT_TOWER
from .common import get_op_tensor_name

__all__ = ['SessionUpdate', 'dump_session_params', 'dump_chkpt_vars',
           'get_savename_from_varname', 'is_training_name',
           'get_checkpoint_path']


def get_savename_from_varname(
        varname, varname_prefix=None,
        savename_prefix=None):
    """
    Args:
        varname(str): a variable name in the graph
        varname_prefix(str): an optional prefix that may need to be removed in varname
        savename_prefix(str): an optional prefix to append to all savename
    Returns:
        str: the name used to save the variable
    """
    name = varname
    if PREDICT_TOWER in name:
        logger.error("No variable under '{}' name scope should be saved!".format(PREDICT_TOWER))
        # don't overwrite anything in the current prediction graph
        return None
    if varname_prefix is not None \
            and name.startswith(varname_prefix):
        name = name[len(varname_prefix) + 1:]
    if savename_prefix is not None:
        name = savename_prefix + '/' + name
    return name


class SessionUpdate(object):
    """ Update the variables in a session """

    def __init__(self, sess, vars_to_update):
        """
        Args:
            sess (tf.Session): a session object
            vars_to_update: a collection of variables to update
        """
        self.sess = sess
        self.name_map = defaultdict(list)
        for v in vars_to_update:
            self.name_map[v.name].append(v)

    @staticmethod
    def load_value_to_var(var, val, strict=False):
        """
        Call `var.load(val)` with the default session.

        Args:
            var (tf.Variable):
            strict (bool): Behave less strict if set to False.
        """
        if strict:
            var.load(val)
            return
        name = var.op.name

        # check incompatible shape
        varshape = tuple(var.get_shape().as_list())
        if varshape != val.shape:
            # TODO only allow reshape when shape different by empty axis
            assert np.prod(varshape) == np.prod(val.shape), \
                "{}: {}!={}".format(name, varshape, val.shape)
            logger.warn("Variable {} is reshaped during assigning".format(name))
            val = val.reshape(varshape)

        # fix some common type incompatibility problem, but is certainly not enough
        def upcast(vartype, valtype):
            # allow up-casting
            if vartype == tf.float64 and valtype == np.float32:
                return np.float64
            if vartype in [tf.int64, tf.int32] and valtype in [np.int32, np.int16, np.int8]:
                return np.int64 if vartype == tf.int64 else np.int32
            return None

        if hasattr(val, 'dtype'):
            vartype = var.value().dtype
            if vartype != val.dtype:
                msg = "Variable {} has dtype {} but was given a value of dtype {}.".format(name, vartype, val.dtype)
                newtype = upcast(var.dtype, val.dtype)
                if newtype is not None:
                    val = newtype(val)
                    logger.warn(msg + " Load it after casting!")
                else:
                    assert vartype == val.dtype, msg
        try:
            var.load(val)
        except tf.errors.InvalidArgumentError:
            logger.exc("Cannot load this value to the variable {}".format(name))

    def update(self, prms):
        """
        Args:
            prms(dict): dict of {variable name: value}
                Any name in prms must be in the graph and in vars_to_update.
        """
        with self.sess.as_default():
            for name, value in six.iteritems(prms):
                assert name in self.name_map
                for v in self.name_map[name]:
                    SessionUpdate.load_value_to_var(v, value)


def dump_session_params(path):
    """
    Dump value of all TRAINABLE + MODEL variables to a dict, and save as
    npy format (loadable by :class:`ParamRestore`).

    Args:
        path(str): the path to save the parameters.
    """
    var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    var.extend(tf.get_collection(tf.GraphKeys.MODEL_VARIABLES))
    # TODO dedup
    assert len(set(var)) == len(var), "TRAINABLE and MODEL variables have duplication!"
    result = {}
    for v in var:
        result[v.name] = v.eval()
    logger.info("Variables to save to {}:".format(path))
    logger.info(str(result.keys()))
    np.save(path, result)


def get_checkpoint_path(model_path):
    """
    Work around TF problems in checkpoint path handling.

    Args:
        model_path: a user-input path
    Returns:
        str: the argument that can be passed to NewCheckpointReader
    """
    if os.path.basename(model_path) == model_path:
        model_path = os.path.join('.', model_path)  # avoid #4921 and #6142
    if os.path.basename(model_path) == 'checkpoint':
        assert os.path.isfile(model_path), model_path
        model_path = tf.train.latest_checkpoint(os.path.dirname(model_path))
        # to be consistent with either v1 or v2

    # fix paths if provided a wrong one
    new_path = model_path
    if '00000-of-00001' in model_path:
        new_path = model_path.split('.data')[0]
    elif model_path.endswith('.index'):
        new_path = model_path.split('.index')[0]
    if new_path != model_path:
        logger.warn(
            "[SaverRestore] {} is corrected to {} when restoring the model.".format(model_path, new_path))
        model_path = new_path
    assert os.path.isfile(model_path) or os.path.isfile(model_path + '.index'), model_path
    return model_path


def dump_chkpt_vars(model_path):
    """ Dump all variables from a checkpoint to a dict.

    Args:
        model_path(str): path to a checkpoint.
    """
    model_path = get_checkpoint_path(model_path)
    reader = tf.train.NewCheckpointReader(model_path)
    var_names = reader.get_variable_to_shape_map().keys()
    result = {}
    for n in var_names:
        result[n] = reader.get_tensor(n)
    return result


def is_training_name(name):
    """
    This is a hack temporarily used to improve logging. Do not use this function.

    Returns:
        bool: Guess whether this tensor is something only used in training.
    """
    # TODO: maybe simply check against TRAINABLE_VARIABLES and MODEL_VARIABLES?
    # TODO or use get_slot_names()
    name = get_op_tensor_name(name)[0]
    if name.endswith('/Adam') or name.endswith('/Adam_1'):
        return True
    if name.endswith('/Momentum'):
        return True
    if name.endswith('/Adadelta') or name.endswith('/Adadelta_1'):
        return True
    if name.endswith('/RMSProp') or name.endswith('/RMSProp_1'):
        return True
    if name.endswith('/Adagrad'):
        return True
    if 'EMA_summary/' in name:
        return True
    return False
