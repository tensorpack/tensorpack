# -*- coding: utf-8 -*-
# File: serialize.py

import os

import pickle
from multiprocessing.reduction import ForkingPickler
import msgpack
import msgpack_numpy

msgpack_numpy.patch()
assert msgpack.version >= (0, 5, 2)

__all__ = ['loads', 'dumps']


MAX_MSGPACK_LEN = 1000000000


class MsgpackSerializer(object):

    @staticmethod
    def dumps(obj):
        """
        Serialize an object.

        Returns:
            Implementation-dependent bytes-like object.
        """
        return msgpack.dumps(obj, use_bin_type=True)

    @staticmethod
    def loads(buf):
        """
        Args:
            buf: the output of `dumps`.
        """
        # Since 0.6, the default max size was set to 1MB.
        # We change it to approximately 1G.
        return msgpack.loads(buf, raw=False,
                             max_bin_len=MAX_MSGPACK_LEN,
                             max_array_len=MAX_MSGPACK_LEN,
                             max_map_len=MAX_MSGPACK_LEN,
                             max_str_len=MAX_MSGPACK_LEN)


class PyarrowSerializer(object):
    @staticmethod
    def dumps(obj):
        """
        Serialize an object.

        Returns:
            Implementation-dependent bytes-like object.
            May not be compatible across different versions of pyarrow.
        """
        import pyarrow as pa
        return pa.serialize(obj).to_buffer()

    @staticmethod
    def dumps_bytes(obj):
        """
        Returns:
            bytes
        """
        return PyarrowSerializer.dumps(obj).to_pybytes()

    @staticmethod
    def loads(buf):
        """
        Args:
            buf: the output of `dumps` or `dumps_bytes`.
        """
        import pyarrow as pa
        return pa.deserialize(buf)


class PickleSerializer(object):
    @staticmethod
    def dumps(obj):
        """
        Returns:
            bytes
        """
        return pickle.dumps(obj, protocol=-1)

    @staticmethod
    def loads(buf):
        """
        Args:
            bytes
        """
        return pickle.loads(buf)


# Define the default serializer to be used that dumps data to bytes
_DEFAULT_S = os.environ.get('TENSORPACK_SERIALIZE', 'pickle')

if _DEFAULT_S == "pyarrow":
    dumps = PyarrowSerializer.dumps_bytes
    loads = PyarrowSerializer.loads
elif _DEFAULT_S == "pickle":
    dumps = PickleSerializer.dumps
    loads = PickleSerializer.loads
else:
    dumps = MsgpackSerializer.dumps
    loads = MsgpackSerializer.loads

# Define the default serializer to be used for passing data
# among a pair of peers. In this case the deserialization is
# known to happen only once
_DEFAULT_S = os.environ.get('TENSORPACK_ONCE_SERIALIZE', 'pickle')

if _DEFAULT_S == "pyarrow":
    dumps_once = PyarrowSerializer.dumps
    loads_once = PyarrowSerializer.loads
elif _DEFAULT_S == "pickle":
    dumps_once = ForkingPickler.dumps
    loads_once = ForkingPickler.loads
else:
    dumps_once = MsgpackSerializer.dumps
    loads_once = MsgpackSerializer.loads
