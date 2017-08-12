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
    zmq_recv, dump_tensor_protos, to_tensor_proto)

try:
    num = int(sys.argv[1])
except:
    num = 2

ENDPOINT = 'ipc://test-pipe'

DATA = []
for k in range(num):
    arr1 = np.random.rand(k + 10, k + 10).astype('float32')
    arr2 = (np.random.rand((k + 10) * 2) * 10).astype('uint8')
    DATA.append([arr1, arr2])


def send():
    ctx = zmq.Context()
    sok = ctx.socket(zmq.PUSH)
    sok.connect(ENDPOINT)

    for arr1, arr2 in DATA:
        t1 = to_tensor_proto(arr1)
        t2 = to_tensor_proto(arr2)
        t = dump_tensor_protos([t1, t2])
        sok.send(t)


def recv():
    sess = tf.InteractiveSession()
    recv = zmq_recv(ENDPOINT, [tf.float32, tf.uint8])
    print(recv)

    for truth in DATA:
        arr = sess.run(recv)
        assert (arr[0] == truth[0]).all()
        assert (arr[1] == truth[1]).all()


p = mp.Process(target=send)
p.start()
recv()
p.join()
