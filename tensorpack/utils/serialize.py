#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: serialize.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import msgpack
import msgpack_numpy

import struct
import numpy as np
from tensorflow.core.framework.tensor_pb2 import TensorProto
from tensorflow.core.framework import types_pb2 as DataType
# have to import like this: https://github.com/tensorflow/tensorflow/commit/955f038afbeb81302cea43058078e68574000bce

msgpack_numpy.patch()


__all__ = ['loads', 'dumps', 'dumps_for_tfop', 'dump_tensor_protos',
           'to_tensor_proto']


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
    np.float32: DataType.DT_FLOAT,
    np.float64: DataType.DT_DOUBLE,
    np.int32: DataType.DT_INT32,
    np.int8: DataType.DT_INT8,
    np.uint8: DataType.DT_UINT8,
}
_DTYPE_DICT = {np.dtype(k): v for k, v in _DTYPE_DICT.items()}


# TODO support string tensor and scalar
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


def dumps_for_tfop(dp):
    protos = [to_tensor_proto(arr) for arr in dp]
    return dump_tensor_protos(protos)
