# -*- coding: UTF-8 -*-
# File: logger.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import logging
import os
import shutil
import os.path
from termcolor import colored
from datetime import datetime
from six.moves import input
import sys

__all__ = ['set_logger_dir', 'disable_logger', 'auto_set_dir']


class _MyFormatter(logging.Formatter):
    def format(self, record):
        date = colored('[%(asctime)s @%(filename)s:%(lineno)d]', 'green')
        msg = '%(message)s'
        if record.levelno == logging.WARNING:
            fmt = date + ' ' + colored('WRN', 'red', attrs=['blink']) + ' ' + msg
        elif record.levelno == logging.ERROR or record.levelno == logging.CRITICAL:
            fmt = date + ' ' + colored('ERR', 'red', attrs=['blink', 'underline']) + ' ' + msg
        else:
            fmt = date + ' ' + msg
        if hasattr(self, '_style'):
            # Python3 compatibilty
            self._style._fmt = fmt
        self._fmt = fmt
        return super(_MyFormatter, self).format(record)


def _getlogger():
    logger = logging.getLogger('tensorpack')
    logger.propagate = False
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_MyFormatter(datefmt='%m%d %H:%M:%S'))
    logger.addHandler(handler)
    return logger


_logger = _getlogger()
_LOGGING_METHOD = ['info', 'warning', 'error', 'critical', 'warn', 'exception', 'debug']
# export logger functions
for func in _LOGGING_METHOD:
    locals()[func] = getattr(_logger, func)


def get_time_str():
    return datetime.now().strftime('%m%d-%H%M%S')


# logger file and directory:
global LOG_FILE, LOG_DIR
LOG_DIR = None


def _set_file(path):
    if os.path.isfile(path):
        backup_name = path + '.' + get_time_str()
        shutil.move(path, backup_name)
        info("Log file '{}' backuped to '{}'".format(path, backup_name))  # noqa: F821
    hdl = logging.FileHandler(
        filename=path, encoding='utf-8', mode='w')
    hdl.setFormatter(_MyFormatter(datefmt='%m%d %H:%M:%S'))
    _logger.addHandler(hdl)
    _logger.info("Argv: " + ' '.join(sys.argv))


def set_logger_dir(dirname, action=None):
    """
    Set the directory for global logging.

    Args:
        dirname(str): log directory
        action(str): an action of ("k","b","d","n") to be performed. Will ask user by default.
    """
    global LOG_FILE, LOG_DIR
    if os.path.isdir(dirname):
        if not action:
            _logger.warn("""\
Directory {} exists! Please either backup/delete it, or use a new directory.""".format(dirname))
            _logger.warn("""\
If you're resuming from a previous run you can choose to keep it.""")
            _logger.info("Select Action: k (keep) / b (backup) / d (delete) / n (new):")
        while not action:
            action = input().lower().strip()
        act = action
        if act == 'b':
            backup_name = dirname + get_time_str()
            shutil.move(dirname, backup_name)
            info("Directory '{}' backuped to '{}'".format(dirname, backup_name))  # noqa: F821
        elif act == 'd':
            shutil.rmtree(dirname)
        elif act == 'n':
            dirname = dirname + get_time_str()
            info("Use a new log directory {}".format(dirname))  # noqa: F821
        elif act == 'k':
            pass
        else:
            raise ValueError("Unknown action: {}".format(act))
    LOG_DIR = dirname
    from .fs import mkdir_p
    mkdir_p(dirname)
    LOG_FILE = os.path.join(dirname, 'log.log')
    _set_file(LOG_FILE)


def disable_logger():
    """ Disable all logging ability from this moment"""
    for func in _LOGGING_METHOD:
        globals()[func] = lambda x: None


def auto_set_dir(action=None, overwrite=False):
    """
    Set log directory to a subdir inside "train_log", with the name being
    the main python file currently running"""
    if LOG_DIR is not None and not overwrite:
        # dir already set
        return
    mod = sys.modules['__main__']
    basename = os.path.basename(mod.__file__)
    set_logger_dir(
        os.path.join('train_log',
                     basename[:basename.rfind('.')]),
        action=action)
