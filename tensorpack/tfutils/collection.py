#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: collection.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import tensorflow as tf
from copy import copy
import six
from contextlib import contextmanager

__all__ = ['backup_collection',
           'restore_collection',
           'clear_collection',
           'freeze_collection']


def backup_collection(keys):
    """
    Args:
        keys (list): list of collection keys to backup
    Returns:
        dict: the backup
    """
    ret = {}
    for k in keys:
        ret[k] = copy(tf.get_collection(k))
    return ret


def restore_collection(backup):
    """
    Restore from a collection backup.

    Args:
        backup (dict):
    """
    for k, v in six.iteritems(backup):
        del tf.get_collection_ref(k)[:]
        tf.get_collection_ref(k).extend(v)


def clear_collection(keys):
    """
    Clear some collections.

    Args:
        keys(list): list of collection keys.
    """
    for k in keys:
        del tf.get_collection_ref(k)[:]


@contextmanager
def freeze_collection(keys):
    """
    Args:
        keys(list): list of collection keys to freeze.

    Returns:
        a context where the collections are in the end restored to its initial state.
    """
    backup = backup_collection(keys)
    yield
    restore_collection(backup)
