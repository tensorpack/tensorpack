# -*- coding: utf-8 -*-
# File: remote.py


import multiprocessing as mp
import time
from collections import deque
import tqdm

from ..utils import logger
from ..utils.concurrency import DIE
from ..utils.serialize import dumps, loads
from ..utils.utils import get_tqdm_kwargs
from .base import DataFlow, DataFlowReentrantGuard

try:
    import zmq
except ImportError:
    logger.warn("Error in 'import zmq'. remote feature won't be available")
    __all__ = []
else:
    __all__ = ['send_dataflow_zmq', 'RemoteDataZMQ']


def send_dataflow_zmq(df, addr, hwm=50, format=None, bind=False):
    """
    Run DataFlow and send data to a ZMQ socket addr.
    It will serialize and send each datapoint to this address with a PUSH socket.
    This function never returns.

    Args:
        df (DataFlow): Will infinitely loop over the DataFlow.
        addr: a ZMQ socket endpoint.
        hwm (int): ZMQ high-water mark (buffer size)
        format (str): The serialization format.
             Default format uses :mod:`utils.serialize`.
             This format works with :class:`dataflow.RemoteDataZMQ`.
             An alternate format is 'zmq_ops', used by https://github.com/tensorpack/zmq_ops
             and :class:`input_source.ZMQInput`.
        bind (bool): whether to bind or connect to the endpoint address.
    """
    assert format in [None, 'zmq_op', 'zmq_ops']
    if format is None:
        dump_fn = dumps
    else:
        from zmq_ops import dump_arrays
        dump_fn = dump_arrays

    ctx = zmq.Context()
    socket = ctx.socket(zmq.PUSH)
    socket.set_hwm(hwm)
    if bind:
        socket.bind(addr)
    else:
        socket.connect(addr)
    try:
        df.reset_state()
        logger.info("Serving data to {} with {} format ...".format(
            addr, 'default' if format is None else 'zmq_ops'))
        INTERVAL = 200
        q = deque(maxlen=INTERVAL)

        try:
            total = len(df)
        except NotImplementedError:
            total = 0
        tqdm_args = get_tqdm_kwargs(leave=True, smoothing=0.8)
        tqdm_args['bar_format'] = tqdm_args['bar_format'] + "{postfix}"
        while True:
            with tqdm.trange(total, **tqdm_args) as pbar:
                for dp in df:
                    start = time.time()
                    socket.send(dump_fn(dp), copy=False)
                    q.append(time.time() - start)
                    pbar.update(1)
                    if pbar.n % INTERVAL == 0:
                        avg = "{:.3f}".format(sum(q) / len(q))
                        pbar.set_postfix({'AvgSendLat': avg})
    finally:
        logger.info("Exiting send_dataflow_zmq ...")
        socket.setsockopt(zmq.LINGER, 0)
        socket.close()
        if not ctx.closed:
            ctx.destroy(0)


class RemoteDataZMQ(DataFlow):
    """
    Produce data from ZMQ PULL socket(s).
    It is the receiver-side counterpart of :func:`send_dataflow_zmq`, which uses :mod:`tensorpack.utils.serialize`
    for serialization.
    See http://tensorpack.readthedocs.io/tutorial/efficient-dataflow.html#distributed-dataflow

    Attributes:
        cnt1, cnt2 (int): number of data points received from addr1 and addr2
    """
    def __init__(self, addr1, addr2=None, hwm=50, bind=True):
        """
        Args:
            addr1,addr2 (str): addr of the zmq endpoint to connect to.
                Use both if you need two protocols (e.g. both IPC and TCP).
                I don't think you'll ever need 3.
            hwm (int): ZMQ high-water mark (buffer size)
            bind (bool): whether to connect or bind the endpoint
        """
        assert addr1
        self._addr1 = addr1
        self._addr2 = addr2
        self._hwm = int(hwm)
        self._guard = DataFlowReentrantGuard()
        self._bind = bind

    def reset_state(self):
        self.cnt1 = 0
        self.cnt2 = 0

    def bind_or_connect(self, socket, addr):
        if self._bind:
            socket.bind(addr)
        else:
            socket.connect(addr)

    def __iter__(self):
        with self._guard:
            try:
                ctx = zmq.Context()
                if self._addr2 is None:
                    socket = ctx.socket(zmq.PULL)
                    socket.set_hwm(self._hwm)
                    self.bind_or_connect(socket, self._addr1)

                    while True:
                        dp = loads(socket.recv(copy=False))
                        yield dp
                        self.cnt1 += 1
                else:
                    socket1 = ctx.socket(zmq.PULL)
                    socket1.set_hwm(self._hwm)
                    self.bind_or_connect(socket1, self._addr1)

                    socket2 = ctx.socket(zmq.PULL)
                    socket2.set_hwm(self._hwm)
                    self.bind_or_connect(socket2, self._addr2)

                    poller = zmq.Poller()
                    poller.register(socket1, zmq.POLLIN)
                    poller.register(socket2, zmq.POLLIN)

                    while True:
                        evts = poller.poll()
                        for sock, evt in evts:
                            dp = loads(sock.recv(copy=False))
                            yield dp
                            if sock == socket1:
                                self.cnt1 += 1
                            else:
                                self.cnt2 += 1
            finally:
                ctx.destroy(linger=0)


# for internal use only
def dump_dataflow_to_process_queue(df, size, nr_consumer):
    """
    Convert a DataFlow to a :class:`multiprocessing.Queue`.
    The DataFlow will only be reset in the spawned process.

    Args:
        df (DataFlow): the DataFlow to dump.
        size (int): size of the queue
        nr_consumer (int): number of consumer of the queue.
            The producer will add this many of ``DIE`` sentinel to the end of the queue.

    Returns:
        tuple(queue, process):
            The process will take data from ``df`` and fill
            the queue, once you start it. Each element in the queue is (idx,
            dp). idx can be the ``DIE`` sentinel when ``df`` is exhausted.
    """
    q = mp.Queue(size)

    class EnqueProc(mp.Process):

        def __init__(self, df, q, nr_consumer):
            super(EnqueProc, self).__init__()
            self.df = df
            self.q = q

        def run(self):
            self.df.reset_state()
            try:
                for idx, dp in enumerate(self.df):
                    self.q.put((idx, dp))
            finally:
                for _ in range(nr_consumer):
                    self.q.put((DIE, None))

    proc = EnqueProc(df, q, nr_consumer)
    return q, proc


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
    # ipc addr like "ipc://@ipc-test"
    if args.task == 'send':
        # use random=True to make it slow and cpu-consuming
        ds = FakeData([(128, 244, 244, 3)], 1000, random=True)
        send_dataflow_zmq(ds, args.addr1)
    else:
        ds = RemoteDataZMQ(args.addr1, args.addr2)
        logger.info("Each DP is 73.5MB")
        TestDataSpeed(ds).start_test()
