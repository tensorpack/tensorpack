# -*- coding: UTF-8 -*-
# File: dftools.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import os
import multiprocessing as mp
from six.moves import range

from .base import DataFlow
from ..utils import logger
from ..utils.utils import get_tqdm
from ..utils.concurrency import DIE
from ..utils.serialize import dumps

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

        # lmdb transaction is not exception-safe!
        # although it has a contextmanager interface
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


def dump_dataflow_to_tfrecord(df, path):
    """
    Dump all datapoints of a Dataflow to a TensorFlow TFRecord file,
    using :func:`serialize.dumps` to serialize.

    Args:
        df (DataFlow):
        path (str): the output file path
    """
    df.reset_state()
    with tf.python_io.TFRecordWriter(path) as writer:
        try:
            sz = df.size()
        except NotImplementedError:
            sz = 0
        with get_tqdm(total=sz) as pbar:
            for dp in df.get_data():
                writer.write(dumps(dp))
                pbar.update()


from ..utils.develop import create_dummy_func  # noqa
try:
    import lmdb
except ImportError:
    dump_dataflow_to_lmdb = create_dummy_func('dump_dataflow_to_lmdb', 'lmdb') # noqa

try:
    import tensorflow as tf
except ImportError:
    dump_dataflow_to_tfrecord = create_dummy_func(  # noqa
        'dump_dataflow_to_tfrecord', 'tensorflow')
