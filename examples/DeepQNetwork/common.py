#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: common.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>
import random
import time
import threading
import multiprocessing
import numpy as np
from tqdm import tqdm
from six.moves import queue

from tensorpack import *
from tensorpack.utils.concurrency import *
from tensorpack.utils.stats import *

global get_player
get_player = None


def play_one_episode(player, func, verbose=False):
    def f(s):
        spc = player.get_action_space()
        act = func([[s]])[0][0].argmax()
        if random.random() < 0.001:
            act = spc.sample()
        if verbose:
            print(act)
        return act
    return np.mean(player.play_one_episode(f))


def play_model(cfg):
    player = get_player(viz=0.01)
    predfunc = OfflinePredictor(cfg)
    while True:
        score = play_one_episode(player, predfunc)
        print("Total:", score)


def eval_with_funcs(predictors, nr_eval):
    class Worker(StoppableThread, ShareSessionThread):
        def __init__(self, func, queue):
            super(Worker, self).__init__()
            self._func = func
            self.q = queue

        def func(self, *args, **kwargs):
            if self.stopped():
                raise RuntimeError("stopped!")
            return self._func(*args, **kwargs)

        def run(self):
            with self.default_sess():
                player = get_player(train=False)
                while not self.stopped():
                    try:
                        score = play_one_episode(player, self.func)
                        # print "Score, ", score
                    except RuntimeError:
                        return
                    self.queue_put_stoppable(self.q, score)

    q = queue.Queue()
    threads = [Worker(f, q) for f in predictors]

    for k in threads:
        k.start()
        time.sleep(0.1)  # avoid simulator bugs
    stat = StatCounter()
    try:
        for _ in tqdm(range(nr_eval), **get_tqdm_kwargs()):
            r = q.get()
            stat.feed(r)
        logger.info("Waiting for all the workers to finish the last run...")
        for k in threads:
            k.stop()
        for k in threads:
            k.join()
        while q.qsize():
            r = q.get()
            stat.feed(r)
    except:
        logger.exception("Eval")
    finally:
        if stat.count > 0:
            return (stat.average, stat.max)
        return (0, 0)


def eval_model_multithread(cfg, nr_eval):
    func = OfflinePredictor(cfg)
    NR_PROC = min(multiprocessing.cpu_count() // 2, 8)
    mean, max = eval_with_funcs([func] * NR_PROC, nr_eval)
    logger.info("Average Score: {}; Max Score: {}".format(mean, max))


class Evaluator(Triggerable):
    def __init__(self, nr_eval, input_names, output_names):
        self.eval_episode = nr_eval
        self.input_names = input_names
        self.output_names = output_names

    def _setup_graph(self):
        NR_PROC = min(multiprocessing.cpu_count() // 2, 20)
        self.pred_funcs = [self.trainer.get_predictor(
            self.input_names, self.output_names)] * NR_PROC

    def _trigger(self):
        t = time.time()
        mean, max = eval_with_funcs(self.pred_funcs, nr_eval=self.eval_episode)
        t = time.time() - t
        if t > 10 * 60:  # eval takes too long
            self.eval_episode = int(self.eval_episode * 0.94)
        self.trainer.monitors.put('mean_score', mean)
        self.trainer.monitors.put('max_score', max)
