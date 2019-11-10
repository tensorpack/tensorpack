# -*- coding: utf-8 -*-
# File: timer.py


import atexit
from collections import defaultdict
from contextlib import contextmanager
from time import perf_counter as timer  # noqa

from . import logger
from .stats import StatCounter


__all__ = ['timed_operation', 'IterSpeedCounter', 'Timer']


@contextmanager
def timed_operation(msg, log_start=False):
    """
    Surround a context with a timer.

    Args:
        msg(str): the log to print.
        log_start(bool): whether to print also at the beginning.

    Example:
        .. code-block:: python

            with timed_operation('Good Stuff'):
                time.sleep(1)

        Will print:

        .. code-block:: python

            Good stuff finished, time:1sec.
    """
    assert len(msg)
    if log_start:
        logger.info('Start {} ...'.format(msg))
    start = timer()
    yield
    msg = msg[0].upper() + msg[1:]
    logger.info('{} finished, time:{:.4f} sec.'.format(
        msg, timer() - start))


_TOTAL_TIMER_DATA = defaultdict(StatCounter)


@contextmanager
def total_timer(msg):
    """ A context which add the time spent inside to the global TotalTimer. """
    start = timer()
    yield
    t = timer() - start
    _TOTAL_TIMER_DATA[msg].feed(t)


def print_total_timer():
    """
    Print the content of the global TotalTimer, if it's not empty. This function will automatically get
    called when program exits.
    """
    if len(_TOTAL_TIMER_DATA) == 0:
        return
    for k, v in _TOTAL_TIMER_DATA.items():
        logger.info("Total Time: {} -> {:.2f} sec, {} times, {:.3g} sec/time".format(
            k, v.sum, v.count, v.average))


atexit.register(print_total_timer)


class IterSpeedCounter(object):
    """ Test how often some code gets reached.

    Example:
        Print the speed of the iteration every 100 times.

        .. code-block:: python

            speed = IterSpeedCounter(100)
            for k in range(1000):
                # do something
                speed()
    """

    def __init__(self, print_every, name=None):
        """
        Args:
            print_every(int): interval to print.
            name(str): name to used when print.
        """
        self.cnt = 0
        self.print_every = int(print_every)
        self.name = name if name else 'IterSpeed'

    def reset(self):
        self.start = timer()

    def __call__(self):
        if self.cnt == 0:
            self.reset()
        self.cnt += 1
        if self.cnt % self.print_every != 0:
            return
        t = timer() - self.start
        logger.info("{}: {:.2f} sec, {} times, {:.3g} sec/time".format(
            self.name, t, self.cnt, t / self.cnt))


class Timer():
    """
    A timer class which computes the time elapsed since the start/reset of the timer.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        """
        Reset the timer.
        """
        self._start = timer()
        self._paused = False
        self._total_paused = 0

    def pause(self):
        """
        Pause the timer.
        """
        assert self._paused is False
        self._paused = timer()

    def is_paused(self):
        return self._paused is not False

    def resume(self):
        """
        Resume the timer.
        """
        assert self._paused is not False
        self._total_paused += timer() - self._paused
        self._paused = False

    def seconds(self):
        """
        Returns:
            float: the total number of seconds since the start/reset of the timer, excluding the
                time in between when the timer is paused.
        """
        if self._paused:
            self.resume()
            self.pause()
        return timer() - self._start - self._total_paused
