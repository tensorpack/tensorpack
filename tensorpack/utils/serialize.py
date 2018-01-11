#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: serialize.py


import msgpack
import msgpack_numpy
msgpack_numpy.patch()

try:
    import pyarrow as pa
except ImportError:
    pass


__all__ = ['loads', 'dumps']


def dumps_msgpack(obj):
    """
    Serialize an object.
    Returns:
        Implementation-dependent bytes-like object
    """
    return msgpack.dumps(obj, use_bin_type=True)


def loads_msgpack(buf):
    """
    Args:
        buf: the output of `dumps`.
    """
    return msgpack.loads(buf, encoding='utf-8')


def dumps_pyarrow(obj):
    """
    Serialize an object.

    Returns:
        Implementation-dependent bytes-like object
    """
    return pa.serialize(obj).to_buffer()


def loads_pyarrow(buf):
    """
    Args:
        buf: the output of `dumps`.
    """
    return pa.deserialize(buf)


loads = loads_msgpack
dumps = dumps_msgpack
