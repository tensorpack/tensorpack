#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: parallel_map.py
import numpy as np
import ctypes
import copy
import threading
import multiprocessing as mp
from six.moves import queue
import zmq

from .base import DataFlow, ProxyDataFlow, DataFlowReentrantGuard
from ..utils.concurrency import StoppableThread
from ..utils import logger
from ..utils.serialize import loads, dumps

from .parallel import (
    _MultiProcessZMQDataFlow, _repeat_iter, _get_pipe_name,
    _bind_guard, _zmq_catch_error)


__all__ = ['ThreadedMapData', 'MultiThreadMapData',
           'MultiProcessMapData', 'MultiProcessMapDataZMQ',
           'MultiProcessMapDataComponentSharedArray']


class _ParallelMapData(ProxyDataFlow):
    def __init__(self, ds, buffer_size):
        super(_ParallelMapData, self).__init__(ds)
        assert buffer_size > 0, buffer_size
        self._buffer_size = buffer_size

    def _recv(self):
        pass

    def _send(self, dp):
        pass

    def _recv_filter_none(self):
        ret = self._recv()
        assert ret is not None, \
            "[{}] Map function cannot return None when strict mode is used.".format(type(self).__name__)
        return ret

    def _fill_buffer(self):
        try:
            for _ in range(self._buffer_size):
                dp = next(self._iter)
                self._send(dp)
        except StopIteration:
            logger.error(
                "[{}] buffer_size cannot be larger than the size of the DataFlow!".format(type(self).__name__))
            raise

    def get_data_non_strict(self):
        for dp in self._iter:
            self._send(dp)
            ret = self._recv()
            if ret is not None:
                yield ret

        self._iter = self.ds.get_data()   # refresh
        for _ in range(self._buffer_size):
            self._send(next(self._iter))
            ret = self._recv()
            if ret is not None:
                yield ret

    def get_data_strict(self):
        for dp in self._iter:
            self._send(dp)
            yield self._recv_filter_none()
        self._iter = self.ds.get_data()   # refresh

        # first clear the buffer, then fill
        for k in range(self._buffer_size):
            dp = self._recv_filter_none()
            if k == self._buffer_size - 1:
                self._fill_buffer()
            yield dp


class MultiThreadMapData(_ParallelMapData):
    """
    Same as :class:`MapData`, but start threads to run the mapping function.
    This is useful when the mapping function is the bottleneck, but you don't
    want to start processes for the entire dataflow pipeline.

    Note:
        1. There is tiny communication overhead with threads, but you
           should avoid starting many threads in your main process to reduce GIL contention.

           The threads will only start in the process which calls :meth:`reset_state()`.
           Therefore you can use ``PrefetchDataZMQ(MultiThreadMapData(...), 1)``
           to reduce GIL contention.

        2. Threads run in parallel and can take different time to run the
           mapping function. Therefore the order of datapoints won't be
           preserved, and datapoints from one pass of `df.get_data()` might get
           mixed with datapoints from the next pass.

           You can use **strict mode**, where `MultiThreadMapData.get_data()`
           is guranteed to produce the exact set which `df.get_data()`
           produces. Although the order of data still isn't preserved.
    """
    class _Worker(StoppableThread):
        def __init__(self, inq, outq, evt, map_func):
            super(MultiThreadMapData._Worker, self).__init__(evt)
            self.inq = inq
            self.outq = outq
            self.func = map_func
            self.daemon = True

        def run(self):
            try:
                while True:
                    dp = self.queue_get_stoppable(self.inq)
                    if self.stopped():
                        return
                    # cannot ignore None here. will lead to unsynced send/recv
                    self.outq.put(self.func(dp))
            except Exception:
                if self.stopped():
                    pass        # skip duplicated error messages
                else:
                    raise
            finally:
                self.stop()

    def __init__(self, ds, nr_thread, map_func, buffer_size=200, strict=False):
        """
        Args:
            ds (DataFlow): the dataflow to map
            nr_thread (int): number of threads to use
            map_func (callable): datapoint -> datapoint | None
            buffer_size (int): number of datapoints in the buffer
            strict (bool): use "strict mode", see notes above.
        """
        super(MultiThreadMapData, self).__init__(ds, buffer_size)

        self._strict = strict
        self.nr_thread = nr_thread
        self.map_func = map_func
        self._threads = []
        self._evt = None

    def reset_state(self):
        super(MultiThreadMapData, self).reset_state()
        if self._threads:
            self._threads[0].stop()
            for t in self._threads:
                t.join()

        self._in_queue = queue.Queue()
        self._out_queue = queue.Queue()
        self._evt = threading.Event()
        self._threads = [MultiThreadMapData._Worker(
            self._in_queue, self._out_queue, self._evt, self.map_func)
            for _ in range(self.nr_thread)]
        for t in self._threads:
            t.start()

        self._iter = self.ds.get_data()
        self._guard = DataFlowReentrantGuard()

        # only call once, to ensure inq+outq has a total of buffer_size elements
        self._fill_buffer()

    def _recv(self):
        return self._out_queue.get()

    def _send(self, dp):
        self._in_queue.put(dp)

    def get_data(self):
        with self._guard:
            if self._strict:
                for dp in self.get_data_strict():
                    yield dp
            else:
                for dp in self.get_data_non_strict():
                    yield dp

    def __del__(self):
        if self._evt is not None:
            self._evt.set()
        for p in self._threads:
            p.join()


# TODO deprecated
ThreadedMapData = MultiThreadMapData


class MultiProcessMapDataZMQ(_ParallelMapData, _MultiProcessZMQDataFlow):
    """
    Same as :class:`MapData`, but start processes to run the mapping function,
    and communicate with ZeroMQ pipe.

    Note:
        1. Processes run in parallel and can take different time to run the
           mapping function. Therefore the order of datapoints won't be
           preserved, and datapoints from one pass of `df.get_data()` might get
           mixed with datapoints from the next pass.

           You can use **strict mode**, where `MultiProcessMapData.get_data()`
           is guranteed to produce the exact set which `df.get_data()`
           produces. Although the order of data still isn't preserved.
    """
    class _Worker(mp.Process):
        def __init__(self, identity, map_func, pipename, hwm):
            super(MultiProcessMapDataZMQ._Worker, self).__init__()
            self.identity = identity
            self.map_func = map_func
            self.pipename = pipename
            self.hwm = hwm

        def run(self):
            ctx = zmq.Context()
            socket = ctx.socket(zmq.DEALER)
            socket.setsockopt(zmq.IDENTITY, self.identity)
            socket.set_hwm(self.hwm)
            socket.connect(self.pipename)

            while True:
                dp = loads(socket.recv(copy=False).bytes)
                dp = self.map_func(dp)
                socket.send(dumps(dp), copy=False)

    def __init__(self, ds, nr_proc, map_func, buffer_size=200, strict=False):
        """
        Args:
            ds (DataFlow): the dataflow to map
            nr_proc(int): number of threads to use
            map_func (callable): datapoint -> datapoint | None
            buffer_size (int): number of datapoints in the buffer
            strict (bool): use "strict mode", see notes above.
        """
        _ParallelMapData.__init__(self, ds, buffer_size)
        _MultiProcessZMQDataFlow.__init__(self)
        self.nr_proc = nr_proc
        self.map_func = map_func
        self._strict = strict
        self._procs = []
        self._guard = DataFlowReentrantGuard()

    def _reset_once(self):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.ROUTER)
        self.socket.set_hwm(self._buffer_size * 2)
        pipename = _get_pipe_name('dataflow-map')
        _bind_guard(self.socket, pipename)

        self._proc_ids = [u'{}'.format(k).encode('utf-8') for k in range(self.nr_proc)]
        worker_hwm = int(self._buffer_size * 2 // self.nr_proc)
        self._procs = [MultiProcessMapDataZMQ._Worker(
            self._proc_ids[k], self.map_func, pipename, worker_hwm)
            for k in range(self.nr_proc)]

        self.ds.reset_state()
        self._iter = self.ds.get_data()
        self._iter_worker = _repeat_iter(lambda: iter(self._proc_ids))

        self._start_processes()
        self._fill_buffer()

    def reset_state(self):
        _MultiProcessZMQDataFlow.reset_state(self)

    def _send(self, dp):
        # round-robin assignment
        worker = next(self._iter_worker)
        msg = [worker, dumps(dp)]
        self.socket.send_multipart(msg, copy=False)

    def _recv(self):
        msg = self.socket.recv_multipart(copy=False)
        dp = loads(msg[1].bytes)
        return dp

    def get_data(self):
        with self._guard, _zmq_catch_error('MultiProcessMapData'):
            if self._strict:
                for dp in self.get_data_strict():
                    yield dp
            else:
                for dp in self.get_data_non_strict():
                    yield dp


MultiProcessMapData = MultiProcessMapDataZMQ  # alias


def _pool_map(data):
    global SHARED_ARR, WORKER_ID, MAP_FUNC
    res = MAP_FUNC(data)
    shared = np.reshape(SHARED_ARR, res.shape)
    assert shared.dtype == res.dtype
    shared[:] = res
    return WORKER_ID


class MultiProcessMapDataComponentSharedArray(DataFlow):
    """
    Similar to :class:`MapDataComponent`, but perform IPC by shared memory,
    therefore more efficient. It requires `map_func` to always return
    a numpy array of fixed shape and dtype, or None.
    """
    def __init__(self, ds, nr_proc, map_func, output_shape, output_dtype, index=0):
        """
        Args:
            ds (DataFlow): the dataflow to map on
            nr_proc(int): number of processes
            map_func (data component -> ndarray | None): the mapping function
            output_shape (tuple): the shape of the output of map_func
            output_dtype (np.dtype): the type of the output of map_func
            index (int): the index of the datapoint component to map on.
        """
        self.ds = ds
        self.nr_proc = nr_proc
        self.map_func = map_func
        self.output_shape = output_shape
        self.output_dtype = np.dtype(output_dtype).type
        self.index = index

        self._shared_mem = [self._create_shared_arr() for k in range(nr_proc)]
        id_queue = mp.Queue()
        for k in range(nr_proc):
            id_queue.put(k)

        def _init_pool(arrs, queue, map_func):
            id = queue.get()
            global SHARED_ARR, WORKER_ID, MAP_FUNC
            SHARED_ARR = arrs[id]
            WORKER_ID = id
            MAP_FUNC = map_func

        self._pool = mp.pool.Pool(
            processes=nr_proc,
            initializer=_init_pool,
            initargs=(self._shared_mem, id_queue, map_func))
        self._guard = DataFlowReentrantGuard()

    def _create_shared_arr(self):
        TYPE = {
            np.float32: ctypes.c_float,
            np.float64: ctypes.c_double,
            np.uint8: ctypes.c_uint8,
            np.int8: ctypes.c_int8,
            np.int32: ctypes.c_int32,
        }
        ctype = TYPE[self.output_dtype]
        arr = mp.RawArray(ctype, int(np.prod(self.output_shape)))
        return arr

    def size(self):
        return self.ds.size()

    def reset_state(self):
        self.ds.reset_state()

    def get_data(self):
        ds_itr = _repeat_iter(self.ds.get_data)
        with self._guard:
            while True:
                dps = []
                for k in range(self.nr_proc):
                    dps.append(copy.copy(next(ds_itr)))
                to_map = [x[self.index] for x in dps]
                res = self._pool.map_async(_pool_map, to_map)

                for index in res.get():
                    arr = np.reshape(self._shared_mem[index], self.output_shape)
                    dp = dps[index]
                    dp[self.index] = arr.copy()
                    yield dp


if __name__ == '__main__':
    class Zero(DataFlow):
        def __init__(self, size):
            self._size = size

        def get_data(self):
            for k in range(self._size):
                yield [k]

        def size(self):
            return self._size

    ds = Zero(300)
    ds = MultiProcessMapData(ds, 3, lambda x: [x[0] + 1], strict=True)
    ds.reset_state()
    for k in ds.get_data():
        print("Bang!", k)
    print("END!")
