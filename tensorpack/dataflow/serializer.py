# -*- coding: utf-8 -*-
# File: serializer.py

import os
import numpy as np
from abc import abstractmethod, ABCMeta
import six

from .base import DataFlow
from ..utils import logger
from ..utils.utils import get_tqdm
from ..utils.serialize import dumps
from .format import LMDBDataPoint, TFRecordDataReader, NumpyDataReader, HDF5Data

__all__ = ['LMDBDataSerializer', 'TFRecordDataSerializer', 'NumpyDataSerializer', 'HDF5DataSerializer']


def delete_file_if_exists(fn):
    try:
        os.remove(fn)
    except OSError:
        pass


@six.add_metaclass(ABCMeta)
class BaseSerializer(object):
    """ Base class for all BaseSerializer

    Remarks:
        This implementation follows SQL related START TRANSACTION and COMMIT Syntax.

    """

    def __init__(self, filename):
        """Summary

        Args:
            df (DataFlow): the DataFlow to dump.
            path (str): the output file path
        """
        self.filename = filename

    def save(self, df, overwrite=False):
        self.save_loop(df, overwrite=overwrite)

    def save_loop(self, df, overwrite=False):
        assert isinstance(df, DataFlow), type(df)
        self.df = df
        if not overwrite:
            assert not os.path.isfile(self.filename)
        else:
            delete_file_if_exists(self.filename)

        try:
            self.size = df.size()
        except NotImplementedError:
            self.size = 0
            logger.warn("Incoming data for serializer has size 0")
        logger.info("begin saving ...")
        self.df.reset_state()
        self._start_transaction()
        with get_tqdm(total=self.size) as pbar:
            for idx, dp in enumerate(self.df.get_data()):
                self._put(idx, dp)
                pbar.update()
        self._commit()
        logger.info("finished saving")

    @abstractmethod
    def _start_transaction(self):
        """ Prepare serialization to disk.
        """

    @abstractmethod
    def _put(self, idx, dp):
        """
        Args:
            idx (int): global index in dataflow (like loop-iterator index)
            dp (list): actual datapoint
        """

    @abstractmethod
    def _commit(self):
        """ Finialize serialization to disk
        """

    @abstractmethod
    def load(self):
        """
        """


class LMDBDataSerializer(BaseSerializer):
    """
    Dump all datapoints of a Dataflow to a LMDB file,
    using :func:`serialize.dumps` to serialize.
    """

    def save(self, df, overwrite=False, write_frequency=5000):
        self.write_frequency = write_frequency
        super(LMDBDataSerializer, self).save(df, overwrite=overwrite)

    def _start_transaction(self):
        self.db = lmdb.open(self.filename, subdir=False,
                            map_size=1099511627776 * 2, readonly=False,
                            meminit=False, map_async=True)    # need sync() at the end
        self.txn = self.db.begin(write=True)
        self.latest_idx = 0

    def _put(self, idx, dp):
        self.txn.put(u'{:08}'.format(idx).encode('ascii'), dumps(dp))
        if (idx + 1) % self.write_frequency == 0:
            self.txn.commit()
            self.txn = self.db.begin(write=True)
        self.latest_idx = idx

    def _commit(self):
        self.txn.commit()
        keys = [u'{:08}'.format(k).encode('ascii') for k in range(self.latest_idx + 1)]
        with self.db.begin(write=True) as txn:
            txn.put(b'__keys__', dumps(keys))

        logger.info("Flushing database ...")
        self.db.sync()
        self.db.close()

    def load(self, shuffle=True):
        return LMDBDataPoint(self.filename, shuffle=shuffle)


class TFRecordDataSerializer(BaseSerializer):
    """
    Dump all datapoints of a Dataflow to a TensorFlow TFRecord file,
    using :func:`serialize.dumps` to serialize.
    """

    def _start_transaction(self):
        self.writer = tf.python_io.TFRecordWriter(self.filename)
        self.writer.__enter__()

    def _put(self, idx, dp):
        self.writer.write(dumps(dp))

    def _commit(self):
        self.writer.__exit__(None, None, None)

    def load(self, size=True):
        return TFRecordDataReader(self.filename, size=size)


class NumpyDataSerializer(BaseSerializer):
    """
    Dump all datapoints of a Dataflow to a Numpy file,
    using :func:`serialize.dumps` to serialize.
    """

    def _start_transaction(self):
        self.buffer = []

    def _put(self, idx, dp):
        self.buffer.append(dumps(dp))

    def _commit(self):
        np.savez_compressed(self.filename, buffer=self.buffer)

    def load(self, shuffle=True):
        return NumpyDataReader(self.filename, shuffle=shuffle)


class HDF5DataSerializer(BaseSerializer):
    """
    Dump all datapoints of a Dataflow to a HDF5 file,
    using :func:`serialize.dumps` to serialize.
    """

    def save(self, df, data_paths, overwrite=False):
        self.data_paths = data_paths
        super(HDF5DataSerializer, self).save(df, overwrite=overwrite)

    def _start_transaction(self):
        self.buffer = dict()
        for data_path in self.data_paths:
            self.buffer[data_path] = []
        self.hf = h5py.File(self.filename, 'w')

    def _put(self, idx, dp):
        assert len(dp) == len(self.data_paths)
        for k, el in zip(self.data_paths, dp):
            self.buffer[k].append(el)

    def _commit(self):
        for data_path in self.data_paths:
            self.hf.create_dataset(data_path, data=self.buffer[data_path])
        self.hf.close()

    def load(self, data_paths, shuffle=True):
        return HDF5Data(self.filename, data_paths, shuffle=shuffle)


from ..utils.develop import create_dummy_func, create_dummy_class  # noqa
try:
    import h5py
except ImportError:
    HDF5DataSerializer = create_dummy_class('HDF5DataSerializer', 'h5py')   # noqa
try:
    import lmdb
except ImportError:
    dump_dataflow_to_lmdb = create_dummy_func('dump_dataflow_to_lmdb', 'lmdb')  # noqa
    LMDBDataSerializer = create_dummy_class('LMDBDataSerializer', 'lmdb')  # noqa

try:
    import tensorflow as tf
except ImportError:
    dump_dataflow_to_tfrecord = create_dummy_func(  # noqa
        'dump_dataflow_to_tfrecord', 'tensorflow')
    TFRecordDataSerializer = create_dummy_class('TFRecordDataSerializer', 'tensorflow')  # noqa
