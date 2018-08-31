# -*- coding: utf-8 -*-
# File: serialize.py

import os
import numpy as np
from collections import defaultdict

from ..utils.utils import get_tqdm
from ..utils import logger
from ..utils.compatible_serialize import dumps, loads

from .base import DataFlow
from .format import LMDBData, HDF5Data
from .common import MapData, FixedSizeData
from .raw import DataFromList, DataFromGenerator

__all__ = ['LMDBSerializer', 'NumpySerializer', 'TFRecordSerializer', 'HDF5Serializer']


def _reset_df_and_get_size(df):
    df.reset_state()
    try:
        sz = len(df)
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
            assert not os.path.isfile(path), "LMDB file {} exists!".format(path)
        db = lmdb.open(path, subdir=isdir,
                       map_size=1099511627776 * 2, readonly=False,
                       meminit=False, map_async=True)    # need sync() at the end
        size = _reset_df_and_get_size(df)
        with get_tqdm(total=size) as pbar:
            idx = -1

            # LMDB transaction is not exception-safe!
            # although it has a context manager interface
            txn = db.begin(write=True)
            for idx, dp in enumerate(df):
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
    and is also >10x slower than LMDB/TFRecord serializers.
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
            for dp in df:
                buffer.append(dp)
                pbar.update()
        np.savez_compressed(path, buffer=np.asarray(buffer, dtype=np.object))

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
        if os.environ.get('TENSORPACK_COMPATIBLE_SERIALIZE', 'msgpack') == 'msgpack':
            def _dumps(dp):
                return dumps(dp)
        else:
            def _dumps(dp):
                return dumps(dp).to_pybytes()

        size = _reset_df_and_get_size(df)
        with tf.python_io.TFRecordWriter(path) as writer, get_tqdm(total=size) as pbar:
            for dp in df:
                writer.write(_dumps(dp))
                pbar.update()

    @staticmethod
    def load(path, size=None):
        """
        Args:
            size (int): total number of records. If not provided, the returned dataflow will have no `__len__()`.
                It's needed because this metadata is not stored in the TFRecord file.
        """
        gen = tf.python_io.tf_record_iterator(path)
        ds = DataFromGenerator(gen)
        ds = MapData(ds, loads)
        if size is not None:
            ds = FixedSizeData(ds, size)
        return ds


class HDF5Serializer():
    """
    Write datapoints to a HDF5 file.

    Note that HDF5 files are in fact not very performant and currently do not support lazy loading.
    It's better to use :class:`LMDBSerializer`.
    """
    @staticmethod
    def save(df, path, data_paths):
        """
        Args:
            df (DataFlow): the DataFlow to serialize.
            path (str): output hdf5 file.
            data_paths (list[str]): list of h5 paths. It should have the same
                length as each datapoint, and each path should correspond to one
                component of the datapoint.
        """
        size = _reset_df_and_get_size(df)
        buffer = defaultdict(list)

        with get_tqdm(total=size) as pbar:
            for dp in df:
                assert len(dp) == len(data_paths), "Datapoint has {} components!".format(len(dp))
                for k, el in zip(data_paths, dp):
                    buffer[k].append(el)
                pbar.update()

        with h5py.File(path, 'w') as hf, get_tqdm(total=len(data_paths)) as pbar:
            for data_path in data_paths:
                hf.create_dataset(data_path, data=buffer[data_path])
                pbar.update()

    @staticmethod
    def load(path, data_paths, shuffle=True):
        """
        Args:
            data_paths (list): list of h5 paths to be zipped.
        """
        return HDF5Data(path, data_paths, shuffle)


from ..utils.develop import create_dummy_class   # noqa
try:
    import lmdb
except ImportError:
    LMDBSerializer = create_dummy_class('LMDBSerializer', 'lmdb')   # noqa

try:
    import tensorflow as tf
except ImportError:
    TFRecordSerializer = create_dummy_class('TFRecordSerializer', 'tensorflow')   # noqa

try:
    import h5py
except ImportError:
    HDF5Serializer = create_dummy_class('HDF5Serializer', 'h5py')   # noqa


if __name__ == '__main__':
    from .raw import FakeData
    import time
    ds = FakeData([[300, 300, 3], [1]], 1000)

    print(time.time())
    TFRecordSerializer.save(ds, 'out.tfrecords')
    print(time.time())
    df = TFRecordSerializer.load('out.tfrecords', size=1000)
    df.reset_state()
    for idx, dp in enumerate(df):
        pass
    print("TF Finished, ", idx)
    print(time.time())

    LMDBSerializer.save(ds, 'out.lmdb')
    print(time.time())
    df = LMDBSerializer.load('out.lmdb')
    df.reset_state()
    for idx, dp in enumerate(df):
        pass
    print("LMDB Finished, ", idx)
    print(time.time())

    NumpySerializer.save(ds, 'out.npz')
    print(time.time())
    df = NumpySerializer.load('out.npz')
    df.reset_state()
    for idx, dp in enumerate(df):
        pass
    print("Numpy Finished, ", idx)
    print(time.time())

    HDF5Serializer.save(ds, 'out.h5')
    print(time.time())
    df = HDF5Serializer.load('out.h5')
    df.reset_state()
    for idx, dp in enumerate(df):
        pass
    print("HDF5 Finished, ", idx)
    print(time.time())
