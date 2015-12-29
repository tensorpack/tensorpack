#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: concurrency.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import threading
from contextlib import contextmanager
from itertools import izip
import tensorflow as tf

from .utils import expand_dim_if_necessary
from .naming import *
import logger

class StoppableThread(threading.Thread):
    def __init__(self):
        super(StoppableThread, self).__init__()
        self._stop = threading.Event()

    def stop(self):
        self._stop.set()

    def stopped(self):
        return self._stop.isSet()


class EnqueueThread(threading.Thread):
    def __init__(self, sess, coord, enqueue_op, dataflow):
        super(EnqueueThread, self).__init__()
        self.sess = sess
        self.coord = coord
        self.input_vars = sess.graph.get_collection(INPUT_VARS_KEY)
        self.dataflow = dataflow
        self.op = enqueue_op
        self.daemon = True

    def run(self):
        try:
            while True:
                for dp in self.dataflow.get_data():
                    if self.coord.should_stop():
                        return
                    feed = dict(izip(self.input_vars, dp))
                    self.sess.run([self.op], feed_dict=feed)
        except tf.errors.CancelledError as e:
            pass
        except Exception:
            # TODO close queue.
            logger.exception("Exception in EnqueueThread:")
            self.coord.request_stop()

@contextmanager
def coordinator_guard(sess, coord):
    try:
        yield
    except (KeyboardInterrupt, Exception) as e:
        raise
    finally:
        coord.request_stop()
        sess.close()
