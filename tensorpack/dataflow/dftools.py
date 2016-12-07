# -*- coding: UTF-8 -*-
# File: dftools.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import sys, os
import cv2
import multiprocessing as mp
import six
from six.moves import range, map

from ..utils import get_tqdm, logger
from ..utils.concurrency import DIE
from ..utils.serialize import dumps
from ..utils.fs import mkdir_p

__all__ = ['dump_dataset_images', 'dataflow_to_process_queue']
try:
    import lmdb
except ImportError:
    logger.warn_dependency("dump_dataflow_to_lmdb", 'lmdb')
else:
    __all__.extend(['dump_dataflow_to_lmdb'])

# TODO pass a name_func to write label as filename?
def dump_dataset_images(ds, dirname, max_count=None, index=0):
    """ Dump images from a `DataFlow` to a directory.

        :param ds: a `DataFlow` instance.
        :param dirname: name of the directory.
        :param max_count: max number of images to dump
        :param index: the index of the image component in a data point.
    """
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
    isdir = os.path.isdir(lmdb_path)
    if isdir:
        assert not os.path.isfile(os.path.join(lmdb_path, 'data.mdb')), "LMDB file exists!"
    else:
        assert not os.path.isfile(lmdb_path), "LMDB file exists!"
    ds.reset_state()
    db = lmdb.open(lmdb_path, subdir=isdir,
            map_size=1099511627776 * 2, readonly=False,
            meminit=False, map_async=True)    # need sync() at the end
    with get_tqdm(total=ds.size()) as pbar:
        with db.begin(write=True) as txn:
            for idx, dp in enumerate(ds.get_data()):
                txn.put(six.binary_type(idx), dumps(dp))
                pbar.update()
            keys = list(map(six.binary_type, range(idx + 1)))
            txn.put('__keys__', dumps(keys))
    logger.info("Flushing database ...")
    db.sync()
    db.close()


def dataflow_to_process_queue(ds, size, nr_consumer):
    """
    Convert a `DataFlow` to a multiprocessing.Queue.
    The dataflow will only be reset in the spawned process.

    :param ds: a `DataFlow`
    :param size: size of the queue
    :param nr_consumer: number of consumer of the queue.
        will add this many of `DIE` sentinel to the end of the queue.
    :returns: (queue, process). The process will take data from `ds` to fill
        the queue once you start it. Each element is (task_id, dp).
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

