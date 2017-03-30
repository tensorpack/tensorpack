#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: remote.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

from ..utils import logger
try:
    import zmq
except ImportError:
    logger.warn("Error in 'import zmq'. remote feature won't be available")
    __all__ = []
else:
    __all__ = ['send_dataflow_zmq', 'RemoteDataZMQ']

from .base import DataFlow
from ..utils import logger
from ..utils.serialize import dumps, loads


def send_dataflow_zmq(df, addr, hwm=50):
    """
    Run DataFlow and send data to a ZMQ socket addr.
    It will dump and send each datapoint to this addr with a PUSH socket.

    Args:
        df (DataFlow): Will infinitely loop over the DataFlow.
        addr: a ZMQ socket addr.
        hwm (int): high water mark
    """
    ctx = zmq.Context()
    socket = ctx.socket(zmq.PUSH)
    socket.set_hwm(hwm)
    socket.connect(addr)
    try:
        df.reset_state()
        logger.info("Serving data to {}".format(addr))
        # TODO print statistics such as speed
        while True:
            for dp in df.get_data():
                socket.send(dumps(dp), copy=False)
    finally:
        socket.setsockopt(zmq.LINGER, 0)
        socket.close()
        if not ctx.closed:
            ctx.destroy(0)


class RemoteDataZMQ(DataFlow):
    """ Produce data from ZMQ PULL socket(s).  """
    def __init__(self, addr1, addr2=None):
        """
        Args:
            addr1,addr2 (str): addr of the socket to connect to.
                Use both if you need two protocols (e.g. both IPC and TCP).
                I don't think you'll ever need 3.
        """
        assert addr1
        self._addr1 = addr1
        self._addr2 = addr2

    def get_data(self):
        try:
            ctx = zmq.Context()
            if self._addr2 is None:
                socket = ctx.socket(zmq.PULL)
                socket.set_hwm(50)
                socket.bind(self._addr1)

                while True:
                    dp = loads(socket.recv(copy=False).bytes)
                    yield dp
            else:
                socket1 = ctx.socket(zmq.PULL)
                socket1.set_hwm(50)
                socket1.bind(self._addr1)

                socket2 = ctx.socket(zmq.PULL)
                socket2.set_hwm(50)
                socket2.bind(self._addr2)

                poller = zmq.Poller()
                poller.register(socket1, zmq.POLLIN)
                poller.register(socket2, zmq.POLLIN)

                while True:
                    evts = poller.poll()
                    for sock, evt in evts:
                        dp = loads(sock.recv(copy=False).bytes)
                        yield dp
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
