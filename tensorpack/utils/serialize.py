# -*- coding: utf-8 -*-
# File: serialize.py

import os
import sys

import msgpack
import msgpack_numpy

from . import logger
from .develop import create_dummy_func

msgpack_numpy.patch()
assert msgpack.version >= (0, 5, 2)

__all__ = ['loads', 'dumps']


MAX_MSGPACK_LEN = 1000000000


def dumps_msgpack(obj):
    """
    Serialize an object.

    Returns:
        Implementation-dependent bytes-like object.
    """
    return msgpack.dumps(obj, use_bin_type=True)


def loads_msgpack(buf):
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


def dumps_pyarrow(obj):
    """
    Serialize an object.

    Returns:
        Implementation-dependent bytes-like object.
        May not be compatible across different versions of pyarrow.
    """
    return pa.serialize(obj).to_buffer()


def loads_pyarrow(buf):
    """
    Args:
        buf: the output of `dumps`.
    """
    return pa.deserialize(buf)


# import pyarrow has a lot of side effect:
# https://github.com/apache/arrow/pull/2329
# https://groups.google.com/a/tensorflow.org/forum/#!topic/developers/TMqRaT-H2bI
# So we use msgpack as default.
if os.environ.get('TENSORPACK_SERIALIZE', 'msgpack') == 'pyarrow':
    try:
        import pyarrow as pa
    except ImportError:
        loads_pyarrow = create_dummy_func('loads_pyarrow', ['pyarrow'])  # noqa
        dumps_pyarrow = create_dummy_func('dumps_pyarrow', ['pyarrow'])  # noqa

    if 'horovod' in sys.modules:
        logger.warn("Horovod and pyarrow may have symbol conflicts. "
                    "Uninstall pyarrow and use msgpack instead.")
    loads = loads_pyarrow
    dumps = dumps_pyarrow
else:
    loads = loads_msgpack
    dumps = dumps_msgpack


class NonPicklableWrapper(object):
    """
    TODO

    https://github.com/joblib/joblib/blob/master/joblib/externals/loky/cloudpickle_wrapper.py
    """
    def __init__(self, obj):
        self._obj = obj

    def __reduce__(self):
        import dill
        s = dill.dumps(self._obj)
        return dill.loads, (s, )

    def __call__(self, *args, **kwargs):
        return self._obj(*args, **kwargs)
