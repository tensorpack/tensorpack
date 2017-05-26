#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: utils.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import copy
from six.moves import zip
from ..tfutils.common import get_op_tensor_name
from ..utils import logger

__all__ = ['get_tensors_inputs', 'get_placeholders_by_names']


def get_tensors_inputs(placeholders, tensors, names):
    """
    Quite often we want to `build_graph()` with normal tensors
    (rather than placeholders).

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


def get_placeholders_by_names(placeholders, names):
    """
    Returns:
        list[Tensor]: a sublist of placeholders, matching names
    """
    placeholder_names = [p.name for p in placeholders]
    ret = []
    for name in names:
        tensorname = get_op_tensor_name(name)[1]
        try:
            idx = placeholder_names.index(tensorname)
        except ValueError:
            logger.error("Name {} is not a model input!".format(tensorname))
            raise
        ret.append(placeholders[idx])
    return ret
