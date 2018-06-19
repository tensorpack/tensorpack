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
           'LMDBDataWriter', 'TfRecordDataWriter', 'NumpyDataWriter', 'HDF5DataWriter',
           'dump_dataflow_to_lmdb_old']


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
    serializer = TfRecordDataWriter(df, path)
    serializer.serialize()


def dump_dataflow_to_lmdb_old(df, lmdb_path, write_frequency=5000):
    """
    JUST FOR HISTORY REASONS TO DEMONSTRATE THIS FUNCTIONS AS WELL DOES NOT PASS THE UNIT TEST
    """
    assert isinstance(df, DataFlow), type(df)
    isdir = os.path.isdir(lmdb_path)
    if isdir:
        assert not os.path.isfile(os.path.join(lmdb_path, 'data.mdb')), "LMDB file exists!"
    else:
        assert not os.path.isfile(lmdb_path), "LMDB file exists!"
    df.reset_state()
    db = lmdb.open(lmdb_path, subdir=isdir,
                   map_size=1099511627776 * 2, readonly=False,
                   meminit=False, map_async=True)    # need sync() at the end
    try:
        sz = df.size()
    except NotImplementedError:
        sz = 0
    with get_tqdm(total=sz) as pbar:
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
    Dump all datapoints of a Dataflow to a TensorFlow TFRecord file,
    using :func:`serialize.dumps` to serialize.
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
        self.txn.put(u'{}'.format(idx).encode('ascii'), dumps(dp))
        if (idx + 1) % self.write_frequency == 0:
            self.txn.commit()
            self.txn = self.db.begin(write=True)
        self.latest_idx = idx

    def _commit(self):
        self.txn.commit()
        keys = [u'{}'.format(k).encode('ascii') for k in range(self.latest_idx + 1)]
        with self.db.begin(write=True) as txn:
            txn.put(b'__keys__', dumps(keys))

        logger.info("Flushing database ...")
        self.db.sync()
        self.db.close()


class TfRecordDataWriter(DataWriter):
    """
    Dump all datapoints of a Dataflow to a TensorFlow TFRecord file,
    using :func:`serialize.dumps` to serialize.
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



from ..utils.develop import create_dummy_func  # noqa
try:
    import h5py
except ImportError:
    HDF5DataWriter = create_dummy_class('HDF5DataWriter', 'h5py')   # noqa
try:
    import lmdb
except ImportError:
    dump_dataflow_to_lmdb = create_dummy_func('dump_dataflow_to_lmdb', 'lmdb') # noqa
    dump_dataflow_to_lmdb_old = create_dummy_func('dump_dataflow_to_lmdb_old', 'lmdb') # noqa
    LMDBDataWriter = create_dummy_class('LMDBDataWriter', 'lmdb') # noqa

try:
    import tensorflow as tf
except ImportError:
    dump_dataflow_to_tfrecord = create_dummy_func(  # noqa
        'dump_dataflow_to_tfrecord', 'tensorflow')
    TfRecordDataWriter = create_dummy_class('TfRecordDataWriter', 'tensorflow') # noqa
