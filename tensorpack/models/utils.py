#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: utils.py

import tensorflow as tf
from ..tfutils.varreplace import custom_getter_scope
from ..tfutils.common import get_tf_version_number
import six


class VariableHolder(object):
    """ A proxy to access variables defined in a layer. """
    def __init__(self, **kwargs):
        """
        Args:
            kwargs: {name:variable}
        """
        self._vars = {}
        for k, v in six.iteritems(kwargs):
            self._add_variable(k, v)

    def _add_variable(self, name, var):
        assert name not in self._vars
        self._vars[name] = var

    def __setattr__(self, name, var):
        if not name.startswith('_'):
            self._add_variable(name, var)
        else:
            # private attributes
            super(VariableHolder, self).__setattr__(name, var)

    def __getattr__(self, name):
        return self._vars[name]

    def all(self):
        """
        Returns:
            list of all variables
        """
        return list(six.itervalues(self._vars))


def rename_get_variable(mapping):
    """
    Args:
        mapping(dict): an old -> new mapping for variable basename. e.g. {'kernel': 'W'}
    """
    def custom_getter(getter, name, *args, **kwargs):
        splits = name.split('/')
        basename = splits[-1]
        if basename in mapping:
            basename = mapping[basename]
            splits[-1] = basename
            name = '/'.join(splits)
        return getter(name, *args, **kwargs)
    return custom_getter_scope(custom_getter)


def monkeypatch_tf_layers():
    if get_tf_version_number() < 1.4:
        if not hasattr(tf.layers, 'Dense'):
            from tensorflow.python.layers.core import Dense
            tf.layers.Dense = Dense

            from tensorflow.python.layers.normalization import BatchNormalization
            tf.layers.BatchNormalization = BatchNormalization

            from tensorflow.python.layers.convolutional import Conv2DTranspose
            tf.layers.Conv2DTranspose = Conv2DTranspose


monkeypatch_tf_layers()
