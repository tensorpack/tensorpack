# -*- coding: utf-8 -*-
# File: varmanip.py

import numpy as np
import os
import pprint
import six
import tensorflow as tf

from ..compat import tfv1
from ..utils import logger
from .common import get_op_tensor_name

__all__ = ['SessionUpdate', 'dump_session_params',
           'load_chkpt_vars', 'save_chkpt_vars', 'get_checkpoint_path']


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
    if varname_prefix is not None \
            and name.startswith(varname_prefix):
        name = name[len(varname_prefix) + 1:]
    if savename_prefix is not None:
        name = savename_prefix + '/' + name
    return name


class SessionUpdate(object):
    """ Update the variables in a session """

    def __init__(self, sess, vars_to_update, ignore_mismatch=False):
        """
        Args:
            sess (tf.Session): a session object
            vars_to_update: a collection of variables to update
            ignore_mismatch (bool): ignore failures when the value and the
                variable does not match.
        """
        self.sess = sess
        self.name_map = {v.name: v for v in vars_to_update}
        self.ignore_mismatch = ignore_mismatch

    @staticmethod
    def relaxed_value_for_var(value, var, ignore_mismatch=False):
        """
        Returns a relaxed (possibly reshaped/upcast-ed) version of value,
        to be loaded to the given variable.

        Args:
            value (ndarray): an numpy array to be loaded to var
            var (tf.Variable):
            ignore_mismatch (bool): ignore failures when the value and the
                variable does not match.

        Returns:
            ndarray: a possibly reshaped or casted version of value.
            Returns None if `ignore_mismatch==True` and the value and the variable
            mismatch.
        """
        assert isinstance(var, tf.Variable)
        name = var.op.name

        # check incompatible shape
        varshape = tuple(var.get_shape().as_list())
        if varshape != value.shape:
            if np.prod(varshape) != np.prod(value.shape):
                if ignore_mismatch:
                    logger.warn(
                        "Cannot load an array of shape {} into variable '{}' whose shape is {}.".format(
                            value.shape, name, varshape))
                    return None
                else:
                    raise ValueError(
                        "Trying to load an array of shape {} into variable '{}' whose shape is {}.".format(
                            value.shape, name, varshape))
            # TODO only allow reshape when shape different by empty axis
            logger.warn("The tensor is reshaped from {} to {} when assigned to '{}'".format(
                value.shape, varshape, name))
            value = value.reshape(varshape)

        # Be permissive, and allow some common type incompatibility problems
        def allow_cast(to_type, from_type):
            # to_type: a tf dtype
            # from_type: a numpy dtype
            from_type = tf.as_dtype(from_type)

            # allow up/down casting between floating points
            if from_type.is_floating and to_type.is_floating:
                return True

            if from_type.is_integer and to_type.is_integer:
                # only allow up-casting between integers
                if to_type.min <= from_type.min and to_type.max >= from_type.max:
                    return True
            return False

        if hasattr(value, 'dtype'):
            vartype = var.dtype.as_numpy_dtype
            if vartype != value.dtype:
                msg = "Variable {} has dtype {} but was given a value of dtype {}.".format(name, var.dtype, value.dtype)

                if allow_cast(var.dtype.base_dtype, value.dtype):
                    value = vartype(value)
                    logger.warn(msg + " The value will be loaded after casting!")
                else:
                    assert vartype == value.dtype, msg
        return value

    def update(self, prms):
        """
        Args:
            prms(dict): dict of {variable name: value}
                Any name in prms must be in the graph and in vars_to_update.
        """
        with self.sess.as_default():
            fetches = []
            feeds = {}
            for name, value in six.iteritems(prms):
                assert name in self.name_map
                var = self.name_map[name]
                value = SessionUpdate.relaxed_value_for_var(
                    value, var, ignore_mismatch=self.ignore_mismatch)
                # This is the implementation of `var.load`
                if value is not None:
                    fetches.append(var.initializer)
                    feeds[var.initializer.inputs[1]] = value
            self.sess.run(fetches, feed_dict=feeds)


def dump_session_params(path):
    """
    Dump value of all TRAINABLE + MODEL variables to a dict, and save as
    npz format (loadable by :func:`sessinit.SmartInit`).

    Args:
        path(str): the file name to save the parameters. Must ends with npz.
    """
    # save variables that are GLOBAL, and either TRAINABLE or MODEL
    var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    var.extend(tf.get_collection(tf.GraphKeys.MODEL_VARIABLES))
    # TODO dedup
    assert len(set(var)) == len(var), "TRAINABLE and MODEL variables have duplication!"
    gvars = {k.name for k in tf.global_variables()}
    var = [v for v in var if v.name in gvars]
    result = {}
    for v in var:
        result[v.name] = v.eval()
    save_chkpt_vars(result, path)


def save_chkpt_vars(dic, path):
    """
    Save variables in dic to path.

    Args:
        dic: {name: value}
        path: save as npz if the name ends with '.npz', otherwise save as a checkpoint.
    """
    logger.info("Variables to save to {}:".format(path))
    keys = sorted(dic.keys())
    logger.info(pprint.pformat(keys))

    assert not path.endswith('.npy')
    if path.endswith('.npz'):
        np.savez_compressed(path, **dic)
    else:
        with tf.Graph().as_default(), \
                tf.Session() as sess:
            for k, v in six.iteritems(dic):
                k = get_op_tensor_name(k)[0]
                _ = tf.Variable(name=k, initial_value=v)    # noqa
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            saver.save(sess, path, write_meta_graph=False)


def get_checkpoint_path(path):
    """
    Work around TF problems in checkpoint path handling.

    Args:
        path: a user-input path
    Returns:
        str: the argument that can be passed to NewCheckpointReader
    """
    if os.path.basename(path) == path:
        path = os.path.join('.', path)  # avoid #4921 and #6142
    if os.path.basename(path) == 'checkpoint':
        assert tfv1.gfile.Exists(path), path
        path = tf.train.latest_checkpoint(os.path.dirname(path))
        # to be consistent with either v1 or v2

    # fix paths if provided a wrong one
    new_path = path
    if '00000-of-00001' in path:
        new_path = path.split('.data')[0]
    elif path.endswith('.index'):
        new_path = path.split('.index')[0]
    if new_path != path:
        logger.info(
            "Checkpoint path {} is auto-corrected to {}.".format(path, new_path))
        path = new_path
    assert tfv1.gfile.Exists(path) or tfv1.gfile.Exists(path + '.index'), path
    return path


def load_chkpt_vars(path):
    """ Load all variables from a checkpoint to a dict.

    Args:
        path(str): path to a checkpoint.

    Returns:
        dict: a name:value dict
    """
    path = get_checkpoint_path(path)
    reader = tfv1.train.NewCheckpointReader(path)
    var_names = reader.get_variable_to_shape_map().keys()
    result = {}
    for n in var_names:
        result[n] = reader.get_tensor(n)
    return result


def is_training_name(name):
    """
    **Guess** if this variable is only used in training.
    Only used internally to avoid too many logging. Do not use it.
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
    if name.startswith('EMA/') or '/EMA/' in name:  # all the moving average summaries
        return True
    if name.startswith('AccumGrad') or name.endswith('/AccumGrad'):
        return True
    if name.startswith('apply_gradients'):
        return True
    return False
