#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: test-recv-op.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import os
import zmq
import argparse
import multiprocessing as mp
import time
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf # noqa
from tensorpack.user_ops.zmq_recv import (  # noqa
    ZMQSocket, dumps_zmq_op)
from tensorpack.utils.concurrency import (  # noqa
    start_proc_mask_signal,
    ensure_proc_terminate)


ENDPOINT = 'ipc://test-pipe'


def send(iterable, delay=0):
    ctx = zmq.Context()
    sok = ctx.socket(zmq.PUSH)
    sok.connect(ENDPOINT)

    for dp in iterable:
        if delay > 0:
            time.sleep(delay)
            print("Sending data to socket..")
        sok.send(dumps_zmq_op(dp))
    time.sleep(999)


def random_array(num):
    ret = []
    for k in range(num):
        arr1 = np.random.rand(k + 10, k + 10).astype('float32')
        # arr1 = 3.0
        arr2 = (np.random.rand((k + 10) * 2) * 10).astype('uint8')
        ret.append([arr1, arr2])
    return ret


def constant_array(num):
    arr = np.ones((30, 30)).astype('float32')
    arr2 = np.ones((3, 3)).astype('uint8')
    return [[arr, arr2]] * num


def hash_dp(dp):
    return sum([k.sum() for k in dp])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', default='basic',
                        choices=['basic', 'tworecv', 'send'])
    parser.add_argument('-n', '--num', type=int, default=10)
    args = parser.parse_args()

    if args.task == 'basic':
        DATA = random_array(args.num)
        p = mp.Process(target=send, args=(DATA,))
        ensure_proc_terminate(p)
        start_proc_mask_signal(p)

        sess = tf.Session()
        recv = ZMQSocket(ENDPOINT, [tf.float32, tf.uint8]).recv()
        print(recv)

        for truth in DATA:
            arr = sess.run(recv)
            assert (arr[0] == truth[0]).all()
            assert (arr[1] == truth[1]).all()
    elif args.task == 'send':
        DATA = random_array(args.num)
        send(DATA)
    elif args.task == 'tworecv':
        DATA = random_array(args.num)
        hashes = [hash_dp(dp) for dp in DATA]
        print(hashes)
        p = mp.Process(target=send, args=(DATA, 0.00))
        ensure_proc_terminate(p)
        start_proc_mask_signal(p)

        sess = tf.Session()
        zmqsock = ZMQSocket(ENDPOINT, [tf.float32, tf.uint8], hwm=1)
        recv1 = zmqsock.recv()
        recv2 = zmqsock.recv()
        print(recv1, recv2)

        for i in range(args.num // 2):
            res1, res2 = sess.run([recv1, recv2])
            h1, h2 = hash_dp(res1), hash_dp(res2)
            print("Recv ", i, h1, h2)
            assert h1 in hashes and h2 in hashes
