#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: test-recv-op.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import sys
import os
import zmq
import multiprocessing as mp
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf # noqa
from tensorpack.user_ops.zmq_recv import (  # noqa
    zmq_recv, dumps_zmq_op)
from tensorpack.utils.concurrency import (  # noqa
    start_proc_mask_signal,
    ensure_proc_terminate)


ENDPOINT = 'ipc://test-pipe'

if __name__ == '__main__':
    try:
        num = int(sys.argv[1])
    except (ValueError, IndexError):
        num = 10

    DATA = []
    for k in range(num):
        arr1 = np.random.rand(k + 10, k + 10).astype('float32')
        arr2 = (np.random.rand((k + 10) * 2) * 10).astype('uint8')
        DATA.append([arr1, arr2])

    def send():
        ctx = zmq.Context()
        sok = ctx.socket(zmq.PUSH)
        sok.connect(ENDPOINT)

        for dp in DATA:
            sok.send(dumps_zmq_op(dp))

    def recv():
        sess = tf.Session()
        recv = zmq_recv(ENDPOINT, [tf.float32, tf.uint8])
        print(recv)

        for truth in DATA:
            arr = sess.run(recv)
            assert (arr[0] == truth[0]).all()
            assert (arr[1] == truth[1]).all()

    p = mp.Process(target=send)
    ensure_proc_terminate(p)
    start_proc_mask_signal(p)
    recv()
    p.join()
