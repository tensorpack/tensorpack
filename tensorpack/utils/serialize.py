#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: serialize.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import pyarrow as pa


__all__ = ['loads', 'dumps']


def dumps(obj):
    """
    Serialize an object.

    Returns:
        Implementation-dependent bytes-like object
    """
    return pa.serialize(obj).to_buffer()


def loads(buf):
    """
    Args:
        buf: the output of `dumps`.
    """
    try:
        return pa.deserialize(buf)
    except pa.ArrowIOError:
        # Handle data serialized by old version of tensorpack.
        import msgpack
        import msgpack_numpy as mn
        return msgpack.unpackb(buf, object_hook=mn.decode, encoding='utf-8')
