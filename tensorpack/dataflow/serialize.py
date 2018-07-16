# -*- coding: utf-8 -*-
# File: serialize.py

import os
import numpy as np

from ..utils.utils import get_tqdm
from ..utils import logger
from ..utils.serialize import dumps, loads

from .base import DataFlow
from .format import LMDBData
from .common import MapData, FixedSizeData
from .raw import DataFromList, DataFromGenerator

__all__ = ['LMDBSerializer', 'NumpySerializer', 'TFRecordSerializer']


def _reset_df_and_get_size(df):
    df.reset_state()
    try:
        sz = df.size()
    except NotImplementedError:
        sz = 0
    return sz


class LMDBSerializer():
    """
    Serialize a Dataflow to a lmdb database, where the keys are indices and values
    are serialized datapoints.

    You will need to `pip install lmdb` to use it.
    """
    @staticmethod
    def save(df, path, write_frequency=5000):
        """
        Args:
            df (DataFlow): the DataFlow to serialize.
            path (str): output path. Either a directory or an lmdb file.
            write_frequency (int): the frequency to write back data to disk.
        """
        assert isinstance(df, DataFlow), type(df)
        isdir = os.path.isdir(path)
        if isdir:
            assert not os.path.isfile(os.path.join(path, 'data.mdb')), "LMDB file exists!"
        else:
            assert not os.path.isfile(path), "LMDB file exists!"
        db = lmdb.open(path, subdir=isdir,
                       map_size=1099511627776 * 2, readonly=False,
                       meminit=False, map_async=True)    # need sync() at the end
        size = _reset_df_and_get_size(df)
        with get_tqdm(total=size) as pbar:
            idx = -1

            # LMDB transaction is not exception-safe!
            # although it has a context manager interface
            txn = db.begin(write=True)
            for idx, dp in enumerate(df.get_data()):
                txn.put(u'{:08}'.format(idx).encode('ascii'), dumps(dp))
                pbar.update()
                if (idx + 1) % write_frequency == 0:
                    txn.commit()
                    txn = db.begin(write=True)
            txn.commit()

            keys = [u'{:08}'.format(k).encode('ascii') for k in range(idx + 1)]
            with db.begin(write=True) as txn:
                txn.put(b'__keys__', dumps(keys))

            logger.info("Flushing database ...")
            db.sync()
        db.close()

    @staticmethod
    def load(path, shuffle=True):
        """
        Note:
            If you found deserialization being the bottleneck, you can use :class:`LMDBData` as the reader
            and run deserialization as a mapper in parallel.
        """
        df = LMDBData(path, shuffle=shuffle)
        return MapData(df, lambda dp: loads(dp[1]))


class NumpySerializer():
    """
    Serialize the entire dataflow to a npz dict.
    Note that this would have to store the entire dataflow in memory,
    and is also >10x slower than the other serializers.
    """
    @staticmethod
    def save(df, path):
        """
        Args:
            df (DataFlow): the DataFlow to serialize.
            path (str): output npz file.
        """
        buffer = []
        size = _reset_df_and_get_size(df)
        with get_tqdm(total=size) as pbar:
            for dp in df.get_data():
                buffer.append(dp)
                pbar.update()
        np.savez_compressed(path, buffer=buffer)

    @staticmethod
    def load(path, shuffle=True):
        buffer = np.load(path)['buffer']
        return DataFromList(buffer, shuffle=shuffle)


class TFRecordSerializer():
    """
    Serialize datapoints to bytes (by tensorpack's default serializer) and write to a TFRecord file.

    Note that TFRecord does not support random access and is in fact not very performant.
    It's better to use :class:`LMDBSerializer`.
    """
    @staticmethod
    def save(df, path):
        """
        Args:
            df (DataFlow): the DataFlow to serialize.
            path (str): output tfrecord file.
        """
        if os.environ.get('TENSORPACK_SERIALIZE', None) == 'msgpack':
            def _dumps(dp):
                return dumps(dp)
        else:
            def _dumps(dp):
                return dumps(dp).to_pybytes()

        size = _reset_df_and_get_size(df)
        with tf.python_io.TFRecordWriter(path) as writer, get_tqdm(total=size) as pbar:
            for dp in df.get_data():
                writer.write(_dumps(dp))
                pbar.update()

    @staticmethod
    def load(path, size=None):
        """
        Args:
            size (int): total number of records. If not provided, the returned dataflow will have no `size()`.
                It's needed because this metadata is not stored in the TFRecord file.
        """
        gen = tf.python_io.tf_record_iterator(path)
        ds = DataFromGenerator(gen)
        ds = MapData(ds, loads)
        if size is not None:
            ds = FixedSizeData(ds, size)
        return ds


from ..utils.develop import create_dummy_class   # noqa
try:
    import lmdb
except ImportError:
    LMDBSerializer = create_dummy_class('LMDBSerializer', 'lmdb')   # noqa

try:
    import tensorflow as tf
except ImportError:
    TFRecordSerializer = create_dummy_class('TFRecordSerializer', 'tensorflow')   # noqa


if __name__ == '__main__':
    from .raw import FakeData
    import time
    ds = FakeData([[300, 300, 3], [1]], 1000)

    print(time.time())
    TFRecordSerializer.save(ds, 'out.tfrecords')
    print(time.time())
    df = TFRecordSerializer.load('out.tfrecords', size=1000)
    df.reset_state()
    for idx, dp in enumerate(df.get_data()):
        pass
    print("TF Finished, ", idx)
    print(time.time())

    LMDBSerializer.save(ds, 'out.lmdb')
    print(time.time())
    df = LMDBSerializer.load('out.lmdb')
    df.reset_state()
    for idx, dp in enumerate(df.get_data()):
        pass
    print("LMDB Finished, ", idx)
    print(time.time())

    NumpySerializer.save(ds, 'out.npz')
    print(time.time())
    df = NumpySerializer.load('out.npz')
    df.reset_state()
    for idx, dp in enumerate(df.get_data()):
        pass
    print("Numpy Finished, ", idx)
    print(time.time())
