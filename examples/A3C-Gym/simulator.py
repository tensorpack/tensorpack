#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: simulator.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import tensorflow as tf
import multiprocessing as mp
import time
import threading
from abc import abstractmethod, ABCMeta
from collections import defaultdict

import six
from six.moves import queue
import zmq

from tensorpack.models.common import disable_layer_logging
from tensorpack.callbacks import Callback
from tensorpack.tfutils.varmanip import SessionUpdate
from tensorpack.predict import OfflinePredictor
from tensorpack.utils import logger
from tensorpack.utils.serialize import loads, dumps
from tensorpack.utils.concurrency import LoopThread, ensure_proc_terminate

__all__ = ['SimulatorProcess', 'SimulatorMaster',
           'SimulatorProcessStateExchange',
           'TransitionExperience']


class TransitionExperience(object):
    """ A transition of state, or experience"""

    def __init__(self, state, action, reward, **kwargs):
        """ kwargs: whatever other attribute you want to save"""
        self.state = state
        self.action = action
        self.reward = reward
        for k, v in six.iteritems(kwargs):
            setattr(self, k, v)


@six.add_metaclass(ABCMeta)
class SimulatorProcessBase(mp.Process):
    def __init__(self, idx):
        super(SimulatorProcessBase, self).__init__()
        self.idx = int(idx)
        self.name = u'simulator-{}'.format(self.idx)
        self.identity = self.name.encode('utf-8')

    @abstractmethod
    def _build_player(self):
        pass


class SimulatorProcessStateExchange(SimulatorProcessBase):
    """
    A process that simulates a player and communicates to master to
    send states and receive the next action
    """

    def __init__(self, idx, pipe_c2s, pipe_s2c):
        """
        :param idx: idx of this process
        """
        super(SimulatorProcessStateExchange, self).__init__(idx)
        self.c2s = pipe_c2s
        self.s2c = pipe_s2c

    def run(self):
        player = self._build_player()
        context = zmq.Context()
        c2s_socket = context.socket(zmq.PUSH)
        c2s_socket.setsockopt(zmq.IDENTITY, self.identity)
        c2s_socket.set_hwm(2)
        c2s_socket.connect(self.c2s)

        s2c_socket = context.socket(zmq.DEALER)
        s2c_socket.setsockopt(zmq.IDENTITY, self.identity)
        # s2c_socket.set_hwm(5)
        s2c_socket.connect(self.s2c)

        state = player.current_state()
        reward, isOver = 0, False
        while True:
            c2s_socket.send(dumps(
                (self.identity, state, reward, isOver)),
                copy=False)
            action = loads(s2c_socket.recv(copy=False).bytes)
            reward, isOver = player.action(action)
            state = player.current_state()


# compatibility
SimulatorProcess = SimulatorProcessStateExchange


class SimulatorMaster(threading.Thread):
    """ A base thread to communicate with all StateExchangeSimulatorProcess.
        It should produce action for each simulator, as well as
        defining callbacks when a transition or an episode is finished.
    """
    class ClientState(object):
        def __init__(self):
            self.memory = []    # list of Experience

    def __init__(self, pipe_c2s, pipe_s2c):
        super(SimulatorMaster, self).__init__()
        self.daemon = True
        self.name = 'SimulatorMaster'

        self.context = zmq.Context()

        self.c2s_socket = self.context.socket(zmq.PULL)
        self.c2s_socket.bind(pipe_c2s)
        self.c2s_socket.set_hwm(10)
        self.s2c_socket = self.context.socket(zmq.ROUTER)
        self.s2c_socket.bind(pipe_s2c)
        self.s2c_socket.set_hwm(10)

        # queueing messages to client
        self.send_queue = queue.Queue(maxsize=100)

        def f():
            msg = self.send_queue.get()
            self.s2c_socket.send_multipart(msg, copy=False)
        self.send_thread = LoopThread(f)
        self.send_thread.daemon = True
        self.send_thread.start()

        # make sure socket get closed at the end
        def clean_context(soks, context):
            for s in soks:
                s.close()
            context.term()
        import atexit
        atexit.register(clean_context, [self.c2s_socket, self.s2c_socket], self.context)

    def run(self):
        self.clients = defaultdict(self.ClientState)
        try:
            while True:
                msg = loads(self.c2s_socket.recv(copy=False).bytes)
                ident, state, reward, isOver = msg
                # TODO check history and warn about dead client
                client = self.clients[ident]

                # check if reward&isOver is valid
                # in the first message, only state is valid
                if len(client.memory) > 0:
                    client.memory[-1].reward = reward
                    if isOver:
                        self._on_episode_over(ident)
                    else:
                        self._on_datapoint(ident)
                # feed state and return action
                self._on_state(state, ident)
        except zmq.ContextTerminated:
            logger.info("[Simulator] Context was terminated.")

    @abstractmethod
    def _on_state(self, state, ident):
        """response to state sent by ident. Preferrably an async call"""

    @abstractmethod
    def _on_episode_over(self, client):
        """ callback when the client just finished an episode.
            You may want to clear the client's memory in this callback.
        """

    def _on_datapoint(self, client):
        """ callback when the client just finished a transition
        """

    def __del__(self):
        self.context.destroy(linger=0)


# ------------------- the following code are not used at all. Just experimental
class SimulatorProcessDF(SimulatorProcessBase):
    """ A simulator which contains a forward model itself, allowing
    it to produce data points directly """

    def __init__(self, idx, pipe_c2s):
        super(SimulatorProcessDF, self).__init__(idx)
        self.pipe_c2s = pipe_c2s

    def run(self):
        self.player = self._build_player()

        self.ctx = zmq.Context()
        self.c2s_socket = self.ctx.socket(zmq.PUSH)
        self.c2s_socket.setsockopt(zmq.IDENTITY, self.identity)
        self.c2s_socket.set_hwm(5)
        self.c2s_socket.connect(self.pipe_c2s)

        self._prepare()
        for dp in self.get_data():
            self.c2s_socket.send(dumps(dp), copy=False)

    @abstractmethod
    def _prepare(self):
        pass

    @abstractmethod
    def get_data(self):
        pass


class SimulatorProcessSharedWeight(SimulatorProcessDF):
    """ A simulator process with an extra thread waiting for event,
    and take shared weight from shm.

    Start me under some CUDA_VISIBLE_DEVICES set!
    """

    def __init__(self, idx, pipe_c2s, condvar, shared_dic, pred_config):
        super(SimulatorProcessSharedWeight, self).__init__(idx, pipe_c2s)
        self.condvar = condvar
        self.shared_dic = shared_dic
        self.pred_config = pred_config

    def _prepare(self):
        disable_layer_logging()
        self.predictor = OfflinePredictor(self.pred_config)
        with self.predictor.graph.as_default():
            vars_to_update = self._params_to_update()
            self.sess_updater = SessionUpdate(
                self.predictor.session, vars_to_update)
        # TODO setup callback for explore?
        self.predictor.graph.finalize()

        self.weight_lock = threading.Lock()

        # start a thread to wait for notification
        def func():
            self.condvar.acquire()
            while True:
                self.condvar.wait()
                self._trigger_evt()
        self.evt_th = threading.Thread(target=func)
        self.evt_th.daemon = True
        self.evt_th.start()

    def _trigger_evt(self):
        with self.weight_lock:
            self.sess_updater.update(self.shared_dic['params'])
            logger.info("Updated.")

    def _params_to_update(self):
        # can be overwritten to update more params
        return tf.trainable_variables()


class WeightSync(Callback):
    """ Sync weight from main process to shared_dic and notify"""

    def __init__(self, condvar, shared_dic):
        self.condvar = condvar
        self.shared_dic = shared_dic

    def _setup_graph(self):
        self.vars = self._params_to_update()

    def _params_to_update(self):
        # can be overwritten to update more params
        return tf.trainable_variables()

    def _before_train(self):
        self._sync()

    def _trigger_epoch(self):
        self._sync()

    def _sync(self):
        logger.info("Updating weights ...")
        dic = {v.name: v.eval() for v in self.vars}
        self.shared_dic['params'] = dic
        self.condvar.acquire()
        self.condvar.notify_all()
        self.condvar.release()


if __name__ == '__main__':
    import random
    from tensorpack.RL import NaiveRLEnvironment

    class NaiveSimulator(SimulatorProcess):

        def _build_player(self):
            return NaiveRLEnvironment()

    class NaiveActioner(SimulatorMaster):
        def _get_action(self, state):
            time.sleep(1)
            return random.randint(1, 12)

        def _on_episode_over(self, client):
            # print("Over: ", client.memory)
            client.memory = []
            client.state = 0

    name = 'ipc://whatever'
    procs = [NaiveSimulator(k, name) for k in range(10)]
    [k.start() for k in procs]

    th = NaiveActioner(name)
    ensure_proc_terminate(procs)
    th.start()

    time.sleep(100)
