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
    __all__ = ['serve_data', 'RemoteData']

from .base import DataFlow
from .common import RepeatedData
from ..utils import logger
from ..utils.serialize import dumps, loads


def serve_data(ds, addr):
    ctx = zmq.Context()
    socket = ctx.socket(zmq.PUSH)
    socket.set_hwm(10)
    socket.bind(addr)
    ds = RepeatedData(ds, -1)
    try:
        ds.reset_state()
        logger.info("Serving data at {}".format(addr))
        # TODO print statistics here
        while True:
            for dp in ds.get_data():
                socket.send(dumps(dp), copy=False)
    finally:
        socket.setsockopt(zmq.LINGER, 0)
        socket.close()
        if not ctx.closed:
            ctx.destroy(0)


class RemoteData(DataFlow):

    def __init__(self, addr):
        self.ctx = zmq.Context()
        self.socket = self.ctx.socket(zmq.PULL)
        self.socket.set_hwm(10)
        self.socket.connect(addr)

    def get_data(self):
        while True:
            dp = loads(self.socket.recv(copy=False))
            yield dp


if __name__ == '__main__':
    import sys
    from tqdm import tqdm
    from .raw import FakeData
    addr = "tcp://127.0.0.1:8877"
    if sys.argv[1] == 'serve':
        ds = FakeData([(128, 244, 244, 3)], 1000)
        serve_data(ds, addr)
    else:
        ds = RemoteData(addr)
        logger.info("Each DP is 73.5MB")
        with tqdm(total=10000) as pbar:
            for k in ds.get_data():
                pbar.update()
