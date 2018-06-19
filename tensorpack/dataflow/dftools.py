# -*- coding: utf-8 -*-
# File: dftools.py


import os
import numpy as np
import multiprocessing as mp
from six.moves import range
from abc import abstractmethod, ABCMeta
import six

from .base import DataFlow
from ..utils import logger
from ..utils.utils import get_tqdm
from ..utils.concurrency import DIE
from ..utils.serialize import dumps

__all__ = ['dump_dataflow_to_process_queue',
           'dump_dataflow_to_lmdb', 'dump_dataflow_to_tfrecord',
           'LMDBDataWriter', 'TFRecordDataWriter', 'NumpyDataWriter', 'HDF5DataWriter']


def dump_dataflow_to_process_queue(df, size, nr_consumer):
    """
    Convert a DataFlow to a :class:`multiprocessing.Queue`.
    The DataFlow will only be reset in the spawned process.

    Args:
        df (DataFlow): the DataFlow to dump.
        size (int): size of the queue
        nr_consumer (int): number of consumer of the queue.
            The producer will add this many of ``DIE`` sentinel to the end of the queue.

    Returns:
        tuple(queue, process):
            The process will take data from ``df`` and fill
            the queue, once you start it. Each element in the queue is (idx,
            dp). idx can be the ``DIE`` sentinel when ``df`` is exhausted.
    """
    q = mp.Queue(size)

    class EnqueProc(mp.Process):

        def __init__(self, df, q, nr_consumer):
            super(EnqueProc, self).__init__()
            self.df = df
            self.q = q

        def run(self):
            self.df.reset_state()
            try:
                for idx, dp in enumerate(self.df.get_data()):
                    self.q.put((idx, dp))
            finally:
                for _ in range(nr_consumer):
                    self.q.put((DIE, None))

    proc = EnqueProc(df, q, nr_consumer)
    return q, proc


def dump_dataflow_to_lmdb(df, lmdb_path, write_frequency=5000):
    """
    Dump a Dataflow to a lmdb database, where the keys are indices and values
    are serialized datapoints.
    The output database can be read directly by
    :class:`tensorpack.dataflow.LMDBDataPoint`.

    Args:
        df (DataFlow): the DataFlow to dump.
        lmdb_path (str): output path. Either a directory or a mdb file.
        write_frequency (int): the frequency to write back data to disk.
    """
    serializer = LMDBDataWriter(df, lmdb_path, write_frequency)
    serializer.serialize()


def dump_dataflow_to_tfrecord(df, path):
    """
    Dump all datapoints of a Dataflow to a TensorFlow TFRecord file,
    using :func:`serialize.dumps` to serialize.
    Args:
        df (DataFlow): the DataFlow to dump.
        path (str): the output file path
    """
    serializer = TFRecordDataWriter(df, path)
    serializer.serialize()


@six.add_metaclass(ABCMeta)
class DataWriter(object):
    """ Base class for all DataWriter"""

    def __init__(self, df, filename):
        """Summary

        Args:
            df (DataFlow): the DataFlow to dump.
            path (str): the output file path
        """
        assert not os.path.isfile(filename)
        self.filename = filename
        assert isinstance(df, DataFlow), type(df)
        self.df = df

        try:
            self.size = df.size()
        except NotImplementedError:
            self.size = 0
            logger.warn("Incoming data for serializer has size 0")

    def begin(self):
        logger.info("begin serializing ...")
        self.df.reset_state()
        self._begin()

    def serialize(self):
        self.begin()
        with get_tqdm(total=self.size) as pbar:
            for idx, dp in enumerate(self.df.get_data()):
                self._put(idx, dp)
                pbar.update()
        self.commit()

    def commit(self):
        self._commit()
        logger.info("finished serializing")

    @abstractmethod
    def _begin(self):
        """
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
        """
        """


class LMDBDataWriter(DataWriter):
    """
    Dump all datapoints of a Dataflow to a LMDB file,
    using :func:`serialize.dumps` to serialize.

    Example:
        .. code-block:: python

            # writing some data
            ds = SomeData()
            LMDBDataWriter(ds, 'test.lmdb').serialize()
            # loading some data
            ds2 = LMDBDataReader('test.lmdb')
    """

    def __init__(self, df, filename, write_frequency=5000):
        """Summary

        Args:
            df (DataFlow): the DataFlow to dump.
            lmdb_path (str): output path. Either a directory or a mdb file.
            write_frequency (int): the frequency to write back data to disk.
        """
        super(LMDBDataWriter, self).__init__(df, filename)
        self.write_frequency = write_frequency

    def _begin(self):
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


class TFRecordDataWriter(DataWriter):
    """
    Dump all datapoints of a Dataflow to a TensorFlow TFRecord file,
    using :func:`serialize.dumps` to serialize.

    Example:
        .. code-block:: python

            # writing some data
            ds = SomeData()
            TFRecordDataWriter(ds, 'test.tfrecord').serialize()
            # loading some data
            ds2 = TFRecordDataReader('test.tfrecord', size=10)
    """
    def _begin(self):
        self.writer = tf.python_io.TFRecordWriter(self.filename)
        self.writer.__enter__()

    def _put(self, idx, dp):
        self.writer.write(dumps(dp))

    def _commit(self):
        self.writer.__exit__(None, None, None)


class NumpyDataWriter(DataWriter):
    """
    Dump all datapoints of a Dataflow to a Numpy file,
    using :func:`serialize.dumps` to serialize.

    Example:
        .. code-block:: python

            # writing some data
            ds = SomeData()
            NumpyDataWriter(ds, 'test.npz').serialize()
            # loading some data
            ds2 = NumpyDataReader('test.npz')
    """
    def _begin(self):
        self.buffer = []

    def _put(self, idx, dp):
        self.buffer.append(dumps(dp))

    def _commit(self):
        np.savez_compressed(self.filename, buffer=self.buffer)


class HDF5DataWriter(DataWriter):
    """
    Dump all datapoints of a Dataflow to a HDF5 file,
    using :func:`serialize.dumps` to serialize.

    Example:
        .. code-block:: python

            # writing some data
            ds = SomeData()
            HDF5DataWriter(ds, 'test.h5', ['label', 'image']).serialize()
            # loading some data
            ds2 = HDF5DataReader('test.h5', ['label', 'image'])
    """

    def __init__(self, df, filename, data_paths):
        """Summary

        Args:
            df (DataFlow): the DataFlow to dump.
            lmdb_path (str): output path. Either a directory or a mdb file.
            write_frequency (int): the frequency to write back data to disk.
        """
        super(HDF5DataWriter, self).__init__(df, filename)
        self.data_paths = data_paths

    def _begin(self):
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



from ..utils.develop import create_dummy_func, create_dummy_class  # noqa
try:
    import h5py
except ImportError:
    HDF5DataWriter = create_dummy_class('HDF5DataWriter', 'h5py')   # noqa
try:
    import lmdb
except ImportError:
    dump_dataflow_to_lmdb = create_dummy_func('dump_dataflow_to_lmdb', 'lmdb') # noqa
    LMDBDataWriter = create_dummy_class('LMDBDataWriter', 'lmdb') # noqa

try:
    import tensorflow as tf
except ImportError:
    dump_dataflow_to_tfrecord = create_dummy_func(  # noqa
        'dump_dataflow_to_tfrecord', 'tensorflow')
    TFRecordDataWriter = create_dummy_class('TFRecordDataWriter', 'tensorflow') # noqa
