# -*- coding: utf-8 -*-
# File: sessinit.py

import os
import numpy as np
import six

from ..compat import tfv1 as tf
from ..utils import logger
from .common import get_op_tensor_name
from .varmanip import SessionUpdate, get_checkpoint_path, get_savename_from_varname, is_training_name

__all__ = ['SessionInit', 'ChainInit',
           'SaverRestore', 'SaverRestoreRelaxed', 'DictRestore',
           'JustCurrentSession', 'get_model_loader', 'SmartInit']


class SessionInit(object):
    """ Base class for utilities to load variables to a (existing) session. """
    def init(self, sess):
        """
        Initialize a session

        Args:
            sess (tf.Session): the session
        """
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
        self._names.append(get_op_tensor_name(name)[0])

    def log(self):
        if len(self._names):
            logger.warn("The following variables are in the {}, but not found in the {}: {}".format(
                self._exists, self._nonexists, ', '.join(self._names)))


class SaverRestore(SessionInit):
    """
    Restore a tensorflow checkpoint saved by :class:`tf.train.Saver` or :class:`ModelSaver`.
    """
    def __init__(self, model_path, prefix=None, ignore=()):
        """
        Args:
            model_path (str): a model name (model-xxxx) or a ``checkpoint`` file.
            prefix (str): during restore, add a ``prefix/`` for every variable in this checkpoint.
            ignore (tuple[str]): tensor names that should be ignored during loading, e.g. learning-rate
        """
        if model_path.endswith('.npy') or model_path.endswith('.npz'):
            logger.warn("SaverRestore expect a TF checkpoint, but got a model path '{}'.".format(model_path) +
                        " To load from a dict, use 'DictRestore'.")
        model_path = get_checkpoint_path(model_path)
        self.path = model_path  # attribute used by AutoResumeTrainConfig!
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
                logger.info("Variable {} in the graph will not be loaded from the checkpoint!".format(name))
            else:
                if reader.has_tensor(name):
                    func(reader, name, v)
                    chkpt_vars_used.add(name)
                else:
                    # use tensor name (instead of op name) for logging, to be consistent with the reverse case
                    if not is_training_name(v.name):
                        mismatch.add(v.name)
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

        When variable shape and value shape do not match, it will print a
        warning but will not crash.

        Another advantage is that it doesn't add any new ops to the graph.
    """
    def _run_init(self, sess):
        logger.info(
            "Restoring checkpoint from {} ...".format(self.path))

        matched_pairs = []

        def f(reader, name, v):
            val = reader.get_tensor(name)
            val = SessionUpdate.relaxed_value_for_var(val, v, ignore_mismatch=True)
            if val is not None:
                matched_pairs.append((v, val))

        with sess.as_default():
            self._match_vars(f)
            upd = SessionUpdate(sess, [x[0] for x in matched_pairs])
            upd.update({x[0].name: x[1] for x in matched_pairs})


class DictRestore(SessionInit):
    """
    Restore variables from a dictionary.
    """

    def __init__(self, variable_dict, ignore_mismatch=False):
        """
        Args:
            variable_dict (dict): a dict of {name: value}
            ignore_mismatch (bool): ignore failures when the value and the
                variable does not match in their shapes.
                If False, it will throw exception on such errors.
                If True, it will only print a warning.
        """
        assert isinstance(variable_dict, dict), type(variable_dict)
        # use varname (with :0) for consistency
        self._prms = {get_op_tensor_name(n)[1]: v for n, v in six.iteritems(variable_dict)}
        self._ignore_mismatch = ignore_mismatch

    def _run_init(self, sess):
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        variable_names_list = [k.name for k in variables]

        variable_names = set(variable_names_list)
        param_names = set(six.iterkeys(self._prms))

        # intersect has the original ordering of variables
        intersect = [v for v in variable_names_list if v in param_names]

        # use opname (without :0) for clarity in logging
        logger.info("Variables to restore from dict: {}".format(
            ', '.join(get_op_tensor_name(x)[0] for x in intersect)))

        mismatch = MismatchLogger('graph', 'dict')
        for k in sorted(variable_names - param_names):
            if not is_training_name(k):
                mismatch.add(k)
        mismatch.log()
        mismatch = MismatchLogger('dict', 'graph')
        for k in sorted(param_names - variable_names):
            mismatch.add(k)
        mismatch.log()

        upd = SessionUpdate(sess, [v for v in variables if v.name in intersect], ignore_mismatch=self._ignore_mismatch)
        logger.info("Restoring {} variables from dict ...".format(len(intersect)))
        upd.update({name: value for name, value in six.iteritems(self._prms) if name in intersect})


class ChainInit(SessionInit):
    """
    Initialize a session by a list of :class:`SessionInit` instance, executed one by one.
    This can be useful for, e.g., loading several models from different files
    to form a composition of models.
    """

    def __init__(self, sess_inits):
        """
        Args:
            sess_inits (list[SessionInit]): list of :class:`SessionInit` instances.
        """
        self.inits = sess_inits

    def _setup_graph(self):
        for i in self.inits:
            i._setup_graph()

    def _run_init(self, sess):
        for i in self.inits:
            i._run_init(sess)


def SmartInit(obj, ignore_mismatch=False):
    """
    Create a :class:`SessionInit` to be loaded to a session,
    automatically from any supported objects, with some smart heuristics.
    The object can be:

    + A TF checkpoint
    + A dict of numpy arrays
    + A npz file, to be interpreted as a dict
    + An empty string or None, in which case the sessinit will be a no-op
    + A list of supported objects, to be initialized one by one

    Args:
        obj: a supported object
        ignore_mismatch (bool): ignore failures when the value and the
            variable does not match in their shapes.
            If False, it will throw exception on such errors.
            If True, it will only print a warning.

    Returns:
        SessionInit:
    """
    if not obj:
        return JustCurrentSession()
    if isinstance(obj, list):
        return ChainInit([SmartInit(x, ignore_mismatch=ignore_mismatch) for x in obj])
    if isinstance(obj, six.string_types):
        obj = os.path.expanduser(obj)
        if obj.endswith(".npy") or obj.endswith(".npz"):
            assert tf.gfile.Exists(obj), "File {} does not exist!".format(obj)
            filename = obj
            logger.info("Loading dictionary from {} ...".format(filename))
            if filename.endswith('.npy'):
                obj = np.load(filename, encoding='latin1').item()
            elif filename.endswith('.npz'):
                obj = dict(np.load(filename))
        elif len(tf.gfile.Glob(obj + "*")):
            # Assume to be a TF checkpoint.
            # A TF checkpoint must be a prefix of an actual file.
            return (SaverRestoreRelaxed if ignore_mismatch else SaverRestore)(obj)
        else:
            raise ValueError("Invalid argument to SmartInit: " + obj)

    if isinstance(obj, dict):
        return DictRestore(obj, ignore_mismatch=ignore_mismatch)
    raise ValueError("Invalid argument to SmartInit: " + type(obj))


get_model_loader = SmartInit
