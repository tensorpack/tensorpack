import os
from .serialize import loads_msgpack, loads_pyarrow, dumps_msgpack, dumps_pyarrow

"""
Serialization that has compatibility guarantee (therefore is safe to store to disk).
"""

__all__ = ['loads', 'dumps']


# pyarrow has no compatibility guarantee
# use msgpack for persistent serialization, unless explicitly set from envvar
if os.environ.get('TENSORPACK_COMPATIBLE_SERIALIZE', 'msgpack') == 'msgpack':
    loads = loads_msgpack
    dumps = dumps_msgpack
else:
    loads = loads_pyarrow
    dumps = dumps_pyarrow
