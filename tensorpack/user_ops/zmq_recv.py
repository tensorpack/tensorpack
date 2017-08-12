#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: zmq_recv.py

import tensorflow as tf
import struct
import numpy as np
import os

from tensorflow.core.framework.tensor_pb2 import TensorProto
from tensorflow.core.framework import types_pb2 as DataType
# have to import like this: https://github.com/tensorflow/tensorflow/commit/955f038afbeb81302cea43058078e68574000bce

from .common import compile

__all__ = ['zmq_recv', 'dumps_for_tfop',
           'dump_tensor_protos', 'to_tensor_proto']


def build():
    global zmq_recv
    ret = compile()
    if ret != 0:
        zmq_recv = None
    else:
        file_dir = os.path.dirname(os.path.abspath(__file__))
        recv_mod = tf.load_op_library(
            os.path.join(file_dir, 'zmq_recv_op.so'))
        zmq_recv = recv_mod.zmq_recv


build()


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
        The format is:

        [#tensors(int32)]
        [tensor1][tensor2]...

        Where each tensor is:

        [dtype(int32)][ndims(int32)][shape[0](int32)]...[shape[n](int32)]
        [len(buffer)(int32)][buffer]
    """
    # TODO use int64

    s = struct.pack('=i', len(protos))
    for p in protos:
        tensor_content = p.tensor_content

        s += struct.pack('=i', int(p.dtype))
        dims = p.tensor_shape.dim
        s += struct.pack('=i', len(dims))
        for k in dims:
            s += struct.pack('=i', k.size)
        s += struct.pack('=i', len(tensor_content))    # won't send stuff over 2G
        s += tensor_content
    return s


def dumps_for_tfop(dp):
    """
    Dump a datapoint (list of nparray) into a format that the ZMQRecv op in tensorpack would accept.
    """
    protos = [to_tensor_proto(arr) for arr in dp]
    return dump_tensor_protos(protos)
