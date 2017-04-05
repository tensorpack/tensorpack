#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: serialize.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import msgpack
import msgpack_numpy

import struct
import numpy as np
from tensorflow.core.framework.tensor_pb2 import TensorProto
# import tensorflow.core.framework.types_pb2 as DataType
from tensorflow.core.framework.types_pb2 import *    # noqa

msgpack_numpy.patch()


__all__ = ['loads', 'dumps']


def dumps(obj):
    """
    Serialize an object.

    Returns:
        str
    """
    return msgpack.dumps(obj, use_bin_type=True)


def loads(buf):
    """
    Args:
        buf (str): serialized object.
    """
    return msgpack.loads(buf)


_DTYPE_DICT = {
    np.float32: DT_FLOAT,   # noqa
    np.float64: DT_DOUBLE,  # noqa
    np.int32: DT_INT32,     # noqa
    np.int8: DT_INT8,       # noqa
    np.uint8: DT_UINT8,     # noqa
}
_DTYPE_DICT = {np.dtype(k): v for k, v in _DTYPE_DICT.items()}


# TODO support string tensor
def to_tensor_proto(arr):
    """
    Convert a numpy array to TensorProto

    Args:
        arr: numpy.ndarray. only supports common numerical types
    """
    dtype = _DTYPE_DICT[arr.dtype]

    ret = TensorProto()
    shape = ret.tensor_shape
    for s in arr.shape:
        d = shape.dim.add()
        d.size = s

    ret.dtype = dtype

    buf = arr.tobytes()
    ret.tensor_content = buf
    return ret


def dump_tensor_protos(protos):
    """
    Serialize a list of :class:`TensorProto`, for communication between custom TensorFlow ops.

    Args:
        protos (list): list of :class:`TensorProto` instance

    Notes:
        The format is: <#protos(int32)>|<size 1>|<serialized proto 1>|<size 2><serialized proto 2>| ...
    """

    s = struct.pack('=i', len(protos))
    for p in protos:
        buf = p.SerializeToString()
        s += struct.pack('=i', len(buf))   # won't send stuff over 2G
        s += buf
    return s
