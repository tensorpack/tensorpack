#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: remote.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import time
import tqdm

from collections import deque
from .base import DataFlow, DataFlowReentrantGuard
from ..utils import logger
from ..utils.utils import get_tqdm_kwargs
from ..utils.serialize import dumps, loads
try:
    import zmq
except ImportError:
    logger.warn("Error in 'import zmq'. remote feature won't be available")
    __all__ = []
else:
    __all__ = ['send_dataflow_zmq', 'RemoteDataZMQ']


def send_dataflow_zmq(df, addr, hwm=50, format=None):
    """
    Run DataFlow and send data to a ZMQ socket addr.
    It will __connect__ to this addr,
    serialize and send each datapoint to this addr with a PUSH socket.
    This function never returns unless an error is encountered.

    Args:
        df (DataFlow): Will infinitely loop over the DataFlow.
        addr: a ZMQ socket endpoint.
        hwm (int): ZMQ high-water mark (buffer size)
        format (str): The serialization format.
             Default format would use :mod:`tensorpack.utils.serialize` (i.e. msgpack).
             An alternate format is 'zmq_op', used by https://github.com/tensorpack/zmq_ops.
    """
    assert format in [None, 'zmq_op']
    if format is None:
        dump_fn = dumps
    else:
        from zmq_ops import dump_arrays
        dump_fn = dump_arrays

    ctx = zmq.Context()
    socket = ctx.socket(zmq.PUSH)
    socket.set_hwm(hwm)
    socket.connect(addr)
    try:
        df.reset_state()
        logger.info("Serving data to {} ...".format(addr))
        INTERVAL = 200
        q = deque(maxlen=INTERVAL)

        try:
            total = df.size()
        except NotImplementedError:
            total = 0
        tqdm_args = get_tqdm_kwargs(leave=True)
        tqdm_args['bar_format'] = tqdm_args['bar_format'] + "{postfix}"
        while True:
            with tqdm.trange(total, **tqdm_args) as pbar:
                for dp in df.get_data():
                    start = time.time()
                    socket.send(dump_fn(dp), copy=False)
                    q.append(time.time() - start)
                    pbar.update(1)
                    if pbar.n % INTERVAL == 0:
                        avg = "{:.3f}".format(sum(q) / len(q))
                        pbar.set_postfix({'AvgSendLat': avg})
    finally:
        socket.setsockopt(zmq.LINGER, 0)
        socket.close()
        if not ctx.closed:
            ctx.destroy(0)


class RemoteDataZMQ(DataFlow):
    """
    Produce data from ZMQ PULL socket(s).
    See http://tensorpack.readthedocs.io/en/latest/tutorial/efficient-dataflow.html#distributed-dataflow

    Attributes:
        cnt1, cnt2 (int): number of data points received from addr1 and addr2
    """
    def __init__(self, addr1, addr2=None, hwm=50):
        """
        Args:
            addr1,addr2 (str): addr of the socket to connect to.
                Use both if you need two protocols (e.g. both IPC and TCP).
                I don't think you'll ever need 3.
            hwm (int): ZMQ high-water mark (buffer size)
        """
        assert addr1
        self._addr1 = addr1
        self._addr2 = addr2
        self._hwm = int(hwm)
        self._guard = DataFlowReentrantGuard()

    def reset_state(self):
        self.cnt1 = 0
        self.cnt2 = 0

    def get_data(self):
        with self._guard:
            try:
                ctx = zmq.Context()
                if self._addr2 is None:
                    socket = ctx.socket(zmq.PULL)
                    socket.set_hwm(self._hwm)
                    socket.bind(self._addr1)

                    while True:
                        dp = loads(socket.recv(copy=False).bytes)
                        yield dp
                        self.cnt1 += 1
                else:
                    socket1 = ctx.socket(zmq.PULL)
                    socket1.set_hwm(self._hwm)
                    socket1.bind(self._addr1)

                    socket2 = ctx.socket(zmq.PULL)
                    socket2.set_hwm(self._hwm)
                    socket2.bind(self._addr2)

                    poller = zmq.Poller()
                    poller.register(socket1, zmq.POLLIN)
                    poller.register(socket2, zmq.POLLIN)

                    while True:
                        evts = poller.poll()
                        for sock, evt in evts:
                            dp = loads(sock.recv(copy=False).bytes)
                            yield dp
                            if sock == socket1:
                                self.cnt1 += 1
                            else:
                                self.cnt2 += 1
            finally:
                ctx.destroy(linger=0)


if __name__ == '__main__':
    from argparse import ArgumentParser
    from .raw import FakeData
    from .common import TestDataSpeed

    """
    Test the multi-producer single-consumer model
    """
    parser = ArgumentParser()
    parser.add_argument('-t', '--task', choices=['send', 'recv'], required=True)
    parser.add_argument('-a', '--addr1', required=True)
    parser.add_argument('-b', '--addr2', default=None)
    args = parser.parse_args()

    # tcp addr like "tcp://127.0.0.1:8877"
    # ipc addr like "ipc:///tmp/ipc-test"
    if args.task == 'send':
        # use random=True to make it slow and cpu-consuming
        ds = FakeData([(128, 244, 244, 3)], 1000, random=True)
        send_dataflow_zmq(ds, args.addr1)
    else:
        ds = RemoteDataZMQ(args.addr1, args.addr2)
        logger.info("Each DP is 73.5MB")
        TestDataSpeed(ds).start_test()
