#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: utils.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import copy
from six.moves import zip
from ..tfutils.common import get_op_tensor_name
from ..utils import logger

__all__ = ['get_tensors_inputs', 'get_sublist_by_names']


def get_tensors_inputs(placeholders, tensors, names):
    """
    Args:
        placeholders (list[Tensor]):
        tensors (list[Tensor]): list of tf.Tensor
        names (list[str]): names matching the tensors

    Returns:
        list[Tensor]: inputs to used with build_graph(),
            with the corresponding placeholders replaced by tensors.
    """
    assert len(tensors) == len(names), \
        "Input tensors {} and input names {} have different length!".format(
            tensors, names)
    ret = copy.copy(placeholders)
    placeholder_names = [p.name for p in placeholders]
    for name, tensor in zip(names, tensors):
        tensorname = get_op_tensor_name(name)[1]
        try:
            idx = placeholder_names.index(tensorname)
        except ValueError:
            logger.error("Name {} is not a model input!".format(tensorname))
            raise
        ret[idx] = tensor
    return ret


def get_sublist_by_names(lst, names):
    """
    Args:
        lst (list): list of objects with "name" property.

    Returns:
        list: a sublist of objects, matching names
    """
    orig_names = [p.name for p in lst]
    ret = []
    for name in names:
        try:
            idx = orig_names.index(name)
        except ValueError:
            logger.error("Name {} doesn't appear in lst {}!".format(
                name, str(orig_names)))
            raise
        ret.append(lst[idx])
    return ret
