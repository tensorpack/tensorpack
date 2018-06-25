# -*- coding: utf-8 -*-
# File: serialize.py

import os
import numpy as np

from ..utils.utils import get_tqdm
from ..utils import logger
from ..utils.serialize import dumps, loads

from .base import DataFlow
from .format import LMDBData, TFRecordData
from .common import MapData
from .raw import DataFromList

__all__ = ['LMDBSerializer', 'NumpySerializer', 'TFRecordSerializer']


def _reset_df_and_get_size(df):
    df.reset_state()
    try:
        sz = df.size()
    except NotImplementedError:
        sz = 0
    return sz


class LMDBSerializer():
    @staticmethod
    def save(df, path, write_frequency=5000):
        """
        Dump a Dataflow to a lmdb database, where the keys are indices and values
        are serialized datapoints.
        The output database can be read directly by
        :class:`tensorpack.dataflow.LMDBDataPoint`.

        Args:
            df (DataFlow): the DataFlow to dump.
            path (str): output path. Either a directory or a mdb file.
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
                txn.put(u'{}'.format(idx).encode('ascii'), dumps(dp))
                pbar.update()
                if (idx + 1) % write_frequency == 0:
                    txn.commit()
                    txn = db.begin(write=True)
            txn.commit()

            keys = [u'{}'.format(k).encode('ascii') for k in range(idx + 1)]
            with db.begin(write=True) as txn:
                txn.put(b'__keys__', dumps(keys))

            logger.info("Flushing database ...")
            db.sync()
        db.close()

    @staticmethod
    def load(path, shuffle=True):
        """
        Args:
            path (str): path to lmdb file
        """
        df = LMDBData(path, shuffle=shuffle)
        return MapData(df, lambda dp: loads(dp[1]))


class NumpySerializer():
    @staticmethod
    def save(df, path):
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
    @staticmethod
    def save(df, path):
        size = _reset_df_and_get_size(df)
        with tf.python_io.TFRecordWriter(path) as writer, get_tqdm(total=size) as pbar:
            for dp in df.get_data():
                writer.write(dumps(dp).to_pybytes())
                pbar.update()

    @staticmethod
    def load(path, size=None):
        return TFRecordData(path, size, decoder=loads)


from ..utils.develop import create_dummy_class   # noqa
try:
    import lmdb
except ImportError:
    LMDBSerializer = create_dummy_class('LMDBSerializer', 'lmdb')   # noqa

try:
    import tensorflow as tf
except ImportError:
    TFRecordSerializer = create_dummy_class('TFRecordSerializer', 'tensorflow')   # noqa
