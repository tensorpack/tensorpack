# -*- coding: UTF-8 -*-
# File: parallel.py


from __future__ import print_function
import weakref
import threading
from contextlib import contextmanager
import multiprocessing as mp
import itertools
from six.moves import range, zip, queue
import errno
import uuid
import os
import zmq
import atexit

from .base import DataFlow, ProxyDataFlow, DataFlowTerminated, DataFlowReentrantGuard
from ..utils.concurrency import (ensure_proc_terminate,
                                 mask_sigint, start_proc_mask_signal,
                                 StoppableThread)
from ..utils.serialize import loads, dumps
from ..utils import logger
from ..utils.gpu import change_gpu

__all__ = ['PrefetchData', 'PrefetchDataZMQ', 'PrefetchOnGPUs',
           'ThreadedMapData', 'MultiThreadMapData',
           'MultiProcessMapData', 'MultiProcessMapDataZMQ']


def _repeat_iter(get_itr):
    while True:
        for x in get_itr():
            yield x


def _bind_guard(sock, name):
    try:
        sock.bind(name)
    except zmq.ZMQError:
        logger.error(
            "ZMQError in socket.bind(). Perhaps you're \
            using pipes on a non-local file system. See documentation of PrefetchDataZMQ for more information.")
        raise


def _get_pipe_name(name):
    pipedir = os.environ.get('TENSORPACK_PIPEDIR', '.')
    assert os.path.isdir(pipedir), pipedir
    pipename = "ipc://{}/{}-pipe-".format(pipedir.rstrip('/'), name) + str(uuid.uuid1())[:6]
    return pipename


def del_weakref(x):
    o = x()
    if o is not None:
        o.__del__()


@contextmanager
def _zmq_catch_error(name):
    try:
        yield
    except zmq.ContextTerminated:
        logger.info("[{}] Context terminated.".format(name))
        raise DataFlowTerminated()
    except zmq.ZMQError as e:
        if e.errno == errno.ENOTSOCK:       # socket closed
            logger.info("[{}] Socket closed.".format(name))
            raise DataFlowTerminated()
        else:
            raise
    except Exception:
        raise


class _MultiProcessZMQDataFlow(DataFlow):
    def __init__(self):
        assert os.name != 'nt', "ZMQ IPC doesn't support windows!"
        self._reset_done = False
        self._procs = []

    def reset_state(self):
        """
        All forked dataflows are reset **once and only once** in spawned processes.
        Nothing more can be done when calling this method.
        """
        if self._reset_done:
            return
        self._reset_done = True

        # __del__ not guranteed to get called at exit
        atexit.register(del_weakref, weakref.ref(self))

        self._reset_once()  # build processes

    def _reset_once(self):
        pass

    def _start_processes(self):
        start_proc_mask_signal(self._procs)

    def __del__(self):
        if not self._reset_done:
            return
        if not self.context.closed:
            self.socket.close(0)
            self.context.destroy(0)
        for x in self._procs:
            x.terminate()
            x.join(5)
        try:
            print("{} successfully cleaned-up.".format(type(self).__name__))
        except Exception:
            pass


class PrefetchData(ProxyDataFlow):
    """
    Prefetch data from a DataFlow using Python multiprocessing utilities.
    It will fork the process calling :meth:`__init__`, collect datapoints from `ds` in each
    process by a Python :class:`multiprocessing.Queue`.

    Note:
        1. An iterator cannot run faster automatically -- what's happenning is
           that the underlying dataflow will be forked ``nr_proc`` times.
           As a result, we have the following guarantee on the dataflow correctness:

           a. When ``nr_proc=1``, the dataflow produces the same data as ``ds`` in the same order.
           b. When ``nr_proc>1``, the dataflow produces the same distribution
              of data as ``ds`` if each sample from ``ds`` is i.i.d. (e.g. fully shuffled).
              You probably only want to use it for training.
        2. This has more serialization overhead than :class:`PrefetchDataZMQ` when data is large.
        3. You can nest like this: ``PrefetchDataZMQ(PrefetchData(df, nr_proc=a), nr_proc=b)``.
           A total of ``a`` instances of ``df`` worker processes will be created.
        4. fork happens in `__init__`. `reset_state()` is a no-op. The worker processes won't get called.
    """

    class _Worker(mp.Process):
        def __init__(self, ds, queue):
            super(PrefetchData._Worker, self).__init__()
            self.ds = ds
            self.queue = queue

        def run(self):
            # reset all ds so each process will produce different data
            self.ds.reset_state()
            while True:
                for dp in self.ds.get_data():
                    self.queue.put(dp)

    def __init__(self, ds, nr_prefetch, nr_proc):
        """
        Args:
            ds (DataFlow): input DataFlow.
            nr_prefetch (int): size of the queue to hold prefetched datapoints.
            nr_proc (int): number of processes to use.
        """
        super(PrefetchData, self).__init__(ds)
        try:
            self._size = ds.size()
        except NotImplementedError:
            self._size = -1
        self.nr_proc = nr_proc
        self.nr_prefetch = nr_prefetch
        self._guard = DataFlowReentrantGuard()

        if nr_proc > 1:
            logger.info("[PrefetchData] Will fork a dataflow more than one times. "
                        "This assumes the datapoints are i.i.d.")

        self.queue = mp.Queue(self.nr_prefetch)
        self.procs = [PrefetchData._Worker(self.ds, self.queue)
                      for _ in range(self.nr_proc)]
        ensure_proc_terminate(self.procs)
        start_proc_mask_signal(self.procs)

    def get_data(self):
        with self._guard:
            for k in itertools.count():
                if self._size > 0 and k >= self._size:
                    break
                dp = self.queue.get()
                yield dp

    def reset_state(self):
        # do nothing. all ds are reset once and only once in spawned processes
        pass


class PrefetchDataZMQ(_MultiProcessZMQDataFlow):
    """
    Prefetch data from a DataFlow using multiple processes, with ZeroMQ for
    communication.
    It will fork the calling process of :meth:`reset_state()`,
    and collect datapoints from the given dataflow in each process by ZeroMQ IPC pipe.

    Note:
        1. An iterator cannot run faster automatically -- what's happenning is
           that the underlying dataflow will be forked ``nr_proc`` times.
           As a result, we have the following guarantee on the dataflow correctness:

           a. When ``nr_proc=1``, this dataflow produces the same data as the
                given dataflow in the same order.
           b. When ``nr_proc>1``, if each sample from the given dataflow is i.i.d. (e.g. fully shuffled),
                then this dataflow produces the **same distribution** of data as the given dataflow.
                This implies that there will be duplication, reordering, etc.
                You probably only want to use it for training.
                If the samples are not i.i.d., the behavior is undefined.
        2. `reset_state()` of the given dataflow will be called **once and only once** in the worker processes.
        3. The fork of processes happened in this dataflow's `reset_state()` method.
           Please note that forking a TensorFlow GPU session may be unsafe.
           If you're managing this dataflow on your own,
           it's better to fork before creating the session.
        4. After the fork has happened, this dataflow becomes not fork-safe.
           i.e., if you fork an already reset instance of this dataflow,
           it won't be usable in the forked process.
        5. Do not nest two `PrefetchDataZMQ`.
        6. By default, a UNIX named pipe will be created in the current directory.
           However, certain non-local filesystem such as NFS/GlusterFS/AFS doesn't always support pipes.
           You can change the directory by ``export TENSORPACK_PIPEDIR=/other/dir``.
           In particular, you can use somewhere under '/tmp' which is usually local.

           Note that some non-local FS may appear to support pipes and code
           may appear to run but crash with bizarre error.
           Also note that ZMQ limits the maximum length of pipe path.
           If you hit the limit, you can set the directory to a softlink
           which points to a local directory.
    """

    class _Worker(mp.Process):
        def __init__(self, ds, conn_name, hwm):
            super(PrefetchDataZMQ._Worker, self).__init__()
            self.ds = ds
            self.conn_name = conn_name
            self.hwm = hwm

        def run(self):
            self.ds.reset_state()
            context = zmq.Context()
            socket = context.socket(zmq.PUSH)
            socket.set_hwm(self.hwm)
            socket.connect(self.conn_name)
            try:
                while True:
                    for dp in self.ds.get_data():
                        socket.send(dumps(dp), copy=False)
            # sigint could still propagate here, e.g. when nested
            except KeyboardInterrupt:
                pass
            finally:
                socket.close(0)
                context.destroy(0)

    def __init__(self, ds, nr_proc=1, hwm=50):
        """
        Args:
            ds (DataFlow): input DataFlow.
            nr_proc (int): number of processes to use.
            hwm (int): the zmq "high-water mark" (queue size) for both sender and receiver.
        """
        super(PrefetchDataZMQ, self).__init__()

        self.ds = ds
        self.nr_proc = nr_proc
        self._hwm = hwm

        self._guard = DataFlowReentrantGuard()
        if nr_proc > 1:
            logger.info("[PrefetchDataZMQ] Will fork a dataflow more than one times. "
                        "This assumes the datapoints are i.i.d.")
        try:
            self._size = ds.size()
        except NotImplementedError:
            self._size = -1

    def _recv(self):
        return loads(self.socket.recv(copy=False).bytes)

    def size(self):
        return self.ds.size()

    def get_data(self):
        with self._guard, _zmq_catch_error('PrefetchDataZMQ'):
            for k in itertools.count():
                if self._size > 0 and k >= self._size:
                    break
                yield self._recv()

    def _reset_once(self):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PULL)
        self.socket.set_hwm(self._hwm)
        pipename = _get_pipe_name('dataflow')
        _bind_guard(self.socket, pipename)

        self._procs = [PrefetchDataZMQ._Worker(self.ds, pipename, self._hwm)
                       for _ in range(self.nr_proc)]
        self._start_processes()


class PrefetchOnGPUs(PrefetchDataZMQ):
    """
    Similar to :class:`PrefetchDataZMQ`,
    but prefetch with each process having its own ``CUDA_VISIBLE_DEVICES`` variable
    mapped to one GPU.
    """

    def __init__(self, ds, gpus):
        """
        Args:
            ds (DataFlow): input DataFlow.
            gpus (list[int]): list of GPUs to use. Will also start this number of processes.
        """
        self.gpus = gpus
        super(PrefetchOnGPUs, self).__init__(ds, len(gpus))

    def _start_processes(self):
        with mask_sigint():
            for gpu, proc in zip(self.gpus, self._procs):
                with change_gpu(gpu):
                    proc.start()


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
