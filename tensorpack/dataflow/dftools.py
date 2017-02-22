# -*- coding: UTF-8 -*-
# File: dftools.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import sys
import os
import cv2
import multiprocessing as mp
import six
from six.moves import range, map

from .base import DataFlow
from ..utils import get_tqdm, logger
from ..utils.concurrency import DIE
from ..utils.serialize import dumps
from ..utils.fs import mkdir_p

__all__ = ['dump_dataset_images', 'dataflow_to_process_queue',
           'dump_dataflow_to_lmdb']


def dump_dataset_images(ds, dirname, max_count=None, index=0):
    """ Dump images from a DataFlow to a directory.

    Args:
        ds (DataFlow): the DataFlow to dump.
        dirname (str): name of the directory.
        max_count (int): limit max number of images to dump. Defaults to unlimited.
        index (int): the index of the image component in the data point.
    """
    # TODO pass a name_func to write label as filename?
    mkdir_p(dirname)
    if max_count is None:
        max_count = sys.maxint
    ds.reset_state()
    for i, dp in enumerate(ds.get_data()):
        if i % 100 == 0:
            print(i)
        if i > max_count:
            return
        img = dp[index]
        cv2.imwrite(os.path.join(dirname, "{}.jpg".format(i)), img)


def dump_dataflow_to_lmdb(ds, lmdb_path):
    """
    Dump a Dataflow to a lmdb database, where the keys are indices and values
    are serialized datapoints.
    The output database can be read directly by
    :class:`tensorpack.dataflow.LMDBDataPoint`.

    Args:
        ds (DataFlow): the DataFlow to dump.
        lmdb_path (str): output path. Either a directory or a mdb file.
    """
    assert isinstance(ds, DataFlow), type(ds)
    isdir = os.path.isdir(lmdb_path)
    if isdir:
        assert not os.path.isfile(os.path.join(lmdb_path, 'data.mdb')), "LMDB file exists!"
    else:
        assert not os.path.isfile(lmdb_path), "LMDB file exists!"
    ds.reset_state()
    db = lmdb.open(lmdb_path, subdir=isdir,
                   map_size=1099511627776 * 2, readonly=False,
                   meminit=False, map_async=True)    # need sync() at the end
    try:
        sz = ds.size()
    except NotImplementedError:
        sz = 0
    with get_tqdm(total=sz) as pbar:
        with db.begin(write=True) as txn:
            for idx, dp in enumerate(ds.get_data()):
                txn.put(six.binary_type(idx), dumps(dp))
                pbar.update()
            keys = list(map(six.binary_type, range(idx + 1)))
            txn.put('__keys__', dumps(keys))

            logger.info("Flushing database ...")
            db.sync()


try:
    import lmdb
except ImportError:
    from ..utils.develop import create_dummy_func
    dump_dataflow_to_lmdb = create_dummy_func('dump_dataflow_to_lmdb', 'lmdb') # noqa


def dataflow_to_process_queue(ds, size, nr_consumer):
    """
    Convert a DataFlow to a :class:`multiprocessing.Queue`.
    The DataFlow will only be reset in the spawned process.

    Args:
        ds (DataFlow): the DataFlow to dump.
        size (int): size of the queue
        nr_consumer (int): number of consumer of the queue.
            The producer will add this many of ``DIE`` sentinel to the end of the queue.

    Returns:
        tuple(queue, process):
            The process will take data from ``ds`` and fill
            the queue, once you start it. Each element in the queue is (idx,
            dp). idx can be the ``DIE`` sentinel when ``ds`` is exhausted.
    """
    q = mp.Queue(size)

    class EnqueProc(mp.Process):

        def __init__(self, ds, q, nr_consumer):
            super(EnqueProc, self).__init__()
            self.ds = ds
            self.q = q

        def run(self):
            self.ds.reset_state()
            try:
                for idx, dp in enumerate(self.ds.get_data()):
                    self.q.put((idx, dp))
            finally:
                for _ in range(nr_consumer):
                    self.q.put((DIE, None))

    proc = EnqueProc(ds, q, nr_consumer)
    return q, proc
