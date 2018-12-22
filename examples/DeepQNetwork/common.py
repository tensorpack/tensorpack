# -*- coding: utf-8 -*-
# File: common.py
# Author: Yuxin Wu
import multiprocessing
import random
import time
from six.moves import queue
from tqdm import tqdm

from tensorpack.callbacks import Callback
from tensorpack.utils import logger
from tensorpack.utils.concurrency import ShareSessionThread, StoppableThread
from tensorpack.utils.stats import StatCounter
from tensorpack.utils.utils import get_tqdm_kwargs


def play_one_episode(env, func, render=False):
    def predict(s):
        """
        Map from observation to action, with 0.01 greedy.
        """
        act = func(s[None, :, :, :])[0][0].argmax()
        if random.random() < 0.01:
            spc = env.action_space
            act = spc.sample()
        return act

    ob = env.reset()
    sum_r = 0
    while True:
        act = predict(ob)
        ob, r, isOver, info = env.step(act)
        if render:
            env.render()
        sum_r += r
        if isOver:
            return sum_r


def play_n_episodes(player, predfunc, nr, render=False):
    logger.info("Start Playing ... ")
    for k in range(nr):
        score = play_one_episode(player, predfunc, render=render)
        print("{}/{}, score={}".format(k, nr, score))


def eval_with_funcs(predictors, nr_eval, get_player_fn, verbose=False):
    """
    Args:
        predictors ([PredictorBase])
    """
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
                player = get_player_fn(train=False)
                while not self.stopped():
                    try:
                        score = play_one_episode(player, self.func)
                    except RuntimeError:
                        return
                    self.queue_put_stoppable(self.q, score)

    q = queue.Queue()
    threads = [Worker(f, q) for f in predictors]

    for k in threads:
        k.start()
        time.sleep(0.1)  # avoid simulator bugs
    stat = StatCounter()

    def fetch():
        r = q.get()
        stat.feed(r)
        if verbose:
            logger.info("Score: {}".format(r))

    for _ in tqdm(range(nr_eval), **get_tqdm_kwargs()):
        fetch()
    # waiting is necessary, otherwise the estimated mean score is biased
    logger.info("Waiting for all the workers to finish the last run...")
    for k in threads:
        k.stop()
    for k in threads:
        k.join()
    while q.qsize():
        fetch()

    if stat.count > 0:
        return (stat.average, stat.max)
    return (0, 0)


def eval_model_multithread(pred, nr_eval, get_player_fn):
    """
    Args:
        pred (OfflinePredictor): state -> [#action]
    """
    NR_PROC = min(multiprocessing.cpu_count() // 2, 8)
    with pred.sess.as_default():
        mean, max = eval_with_funcs(
            [pred] * NR_PROC, nr_eval,
            get_player_fn, verbose=True)
    logger.info("Average Score: {}; Max Score: {}".format(mean, max))


class Evaluator(Callback):
    def __init__(self, nr_eval, input_names, output_names, get_player_fn):
        self.eval_episode = nr_eval
        self.input_names = input_names
        self.output_names = output_names
        self.get_player_fn = get_player_fn

    def _setup_graph(self):
        NR_PROC = min(multiprocessing.cpu_count() // 2, 20)
        self.pred_funcs = [self.trainer.get_predictor(
            self.input_names, self.output_names)] * NR_PROC

    def _trigger(self):
        t = time.time()
        mean, max = eval_with_funcs(
            self.pred_funcs, self.eval_episode, self.get_player_fn)
        t = time.time() - t
        if t > 10 * 60:  # eval takes too long
            self.eval_episode = int(self.eval_episode * 0.94)
        self.trainer.monitors.put_scalar('mean_score', mean)
        self.trainer.monitors.put_scalar('max_score', max)
