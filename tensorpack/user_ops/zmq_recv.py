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

from .common import compile, get_ext_suffix

__all__ = ['dumps_zmq_op', 'ZMQRecv',
           'dump_tensor_protos', 'to_tensor_proto']


_zmq_recv_mod = None


def try_build():
    file_dir = os.path.dirname(os.path.abspath(__file__))
    basename = 'zmq_recv_op' + get_ext_suffix()
    so_file = os.path.join(file_dir, basename)
    if not os.path.isfile(so_file):
        ret = compile()
        if ret != 0:
            raise RuntimeError("tensorpack user_ops compilation failed!")

    global _zmq_recv_mod
    _zmq_recv_mod = tf.load_op_library(so_file)


try_build()


class ZMQRecv(object):
    def __init__(self, end_point, types, hwm=None, name=None):
        self._types = types

        if name is None:
            self._name = (tf.get_default_graph()
                          .unique_name(self.__class__.__name__))
        else:
            self._name = name

        self._zmq_handle = _zmq_recv_mod.zmq_connection(
            end_point, hwm, shared_name=self._name)

    @property
    def name(self):
        return self._name

    def recv(self):
        return _zmq_recv_mod.zmq_recv(
            self._zmq_handle, self._types)


_DTYPE_DICT = {
    np.float32: DataType.DT_FLOAT,
    np.float64: DataType.DT_DOUBLE,
    np.int32: DataType.DT_INT32,
    np.int8: DataType.DT_INT8,
    np.uint8: DataType.DT_UINT8,
}
_DTYPE_DICT = {np.dtype(k): v for k, v in _DTYPE_DICT.items()}


def to_tensor_proto(arr):
    """
    Convert a numpy array to TensorProto

    Args:
        arr: numpy.ndarray. only supports common numerical types
    """
    if isinstance(arr, float):
        arr = np.asarray(arr).astype('float32')
    elif isinstance(arr, int):
        arr = np.asarray(arr).astype('int32')
    assert isinstance(arr, np.ndarray), type(arr)
    try:
        dtype = _DTYPE_DICT[arr.dtype]
    except KeyError:
        raise KeyError("Dtype {} is unsupported by current ZMQ Op!".format(arr.dtype))

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
        [len(buffer)(int64)][buffer]
    """

    s = struct.pack('=i', len(protos))
    for p in protos:
        tensor_content = p.tensor_content

        s += struct.pack('=i', int(p.dtype))
        dims = p.tensor_shape.dim
        s += struct.pack('=i', len(dims))
        for k in dims:
            s += struct.pack('=i', k.size)
        s += struct.pack('=q', len(tensor_content))
        s += tensor_content
    return s


def dumps_zmq_op(dp):
    """
    Dump a datapoint (list of nparray) into a format that the ZMQRecv op in tensorpack would accept.

    Args:
        dp: list of nparray

    Returns:
        a binary string
    """
    assert isinstance(dp, (list, tuple))
    protos = [to_tensor_proto(arr) for arr in dp]
    return dump_tensor_protos(protos)
