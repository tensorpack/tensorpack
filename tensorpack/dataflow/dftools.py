# -*- coding: utf-8 -*-
# File: dftools.py


import multiprocessing as mp
from six.moves import range

from ..utils.concurrency import DIE

from .serialize import LMDBSerializer, TFRecordSerializer

__all__ = ['dump_dataflow_to_process_queue',
           'dump_dataflow_to_lmdb', 'dump_dataflow_to_tfrecord']


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
    # TODO mark deprecated
    LMDBSerializer.save(df, lmdb_path, write_frequency)


def dump_dataflow_to_tfrecord(df, path):
    """
    Dump all datapoints of a Dataflow to a TensorFlow TFRecord file,
    using :func:`serialize.dumps` to serialize.

    Args:
        df (DataFlow):
        path (str): the output file path
    """
    # TODO mark deprecated
    TFRecordSerializer.save(df, path)
