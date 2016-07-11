# -*- coding: UTF-8 -*-
# File: dftools.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import sys, os
import cv2
import multiprocessing

from ..utils.concurrency import DIE
from ..utils.fs import mkdir_p

__all__ = ['dump_dataset_images', 'dataflow_to_process_queue']

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
    for i, dp in enumerate(ds.get_data()):
        if i % 100 == 0:
            print(i)
        if i > max_count:
            return
        img = dp[index]
        cv2.imwrite(os.path.join(dirname, "{}.jpg".format(i)), img)

def dataflow_to_process_queue(ds, size, nr_consumer):
    """
    Convert a `DataFlow` to a multiprocessing.Queue.

    :param ds: a `DataFlow`
    :param size: size of the queue
    :param nr_consumer: number of consumer of the queue.
        will add this many of `DIE` sentinel to the end of the queue.
    :returns: (queue, process). The process will take data from `ds` to fill
        the queue once you start it. Each element is (task_id, dp).
    """
    q = multiprocessing.Queue(size)
    class EnqueProc(multiprocessing.Process):
        def __init__(self, ds, q, nr_consumer):
            super(EnqueProc, self).__init__()
            self.ds = ds
            self.q = q

        def run(self):
            try:
                for idx, dp in enumerate(self.ds.get_data()):
                    self.q.put((idx, dp))
            finally:
                for _ in range(nr_consumer):
                    self.q.put((DIE, None))

    proc = EnqueProc(ds, q, nr_consumer)
    return q, proc


