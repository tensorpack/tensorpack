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

__all__ = ['set_logger_dir', 'auto_set_dir']


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
_LOGGING_METHOD = ['info', 'warning', 'error', 'critical', 'warn', 'exception', 'debug', 'setLevel']
# export logger functions
for func in _LOGGING_METHOD:
    locals()[func] = getattr(_logger, func)
    __all__.append(func)


def _get_time_str():
    return datetime.now().strftime('%m%d-%H%M%S')


# logger file and directory:
global LOG_DIR, _FILE_HANDLER
LOG_DIR = None
_FILE_HANDLER = None


def _set_file(path):
    global _FILE_HANDLER
    if os.path.isfile(path):
        backup_name = path + '.' + _get_time_str()
        shutil.move(path, backup_name)
        info("Log file '{}' backuped to '{}'".format(path, backup_name))  # noqa: F821
    hdl = logging.FileHandler(
        filename=path, encoding='utf-8', mode='w')
    hdl.setFormatter(_MyFormatter(datefmt='%m%d %H:%M:%S'))

    _FILE_HANDLER = hdl
    _logger.addHandler(hdl)
    _logger.info("Argv: " + ' '.join(sys.argv))


def set_logger_dir(dirname, action=None):
    """
    Set the directory for global logging.

    Args:
        dirname(str): log directory
        action(str): an action of ("k","b","d","n","q") to be performed. Will ask user by default.
    """
    global LOG_DIR, _FILE_HANDLER
    if _FILE_HANDLER:
        # unload and close the old file handler, so that we may safely delete the logger directory
        _logger.removeHandler(_FILE_HANDLER)
        del _FILE_HANDLER
    if os.path.isdir(dirname):
        if not action:
            _logger.warn("""\
Log directory {} exists! Please either backup/delete it, or use a new directory.""".format(dirname))
            _logger.warn("""\
If you're resuming from a previous run you can choose to keep it.""")
            _logger.info("Select Action: k (keep) / b (backup) / d (delete) / n (new) / q (quit):")
        while not action:
            action = input().lower().strip()
        act = action
        if act == 'b':
            backup_name = dirname + _get_time_str()
            shutil.move(dirname, backup_name)
            info("Directory '{}' backuped to '{}'".format(dirname, backup_name))  # noqa: F821
        elif act == 'd':
            shutil.rmtree(dirname)
        elif act == 'n':
            dirname = dirname + _get_time_str()
            info("Use a new log directory {}".format(dirname))  # noqa: F821
        elif act == 'k':
            pass
        elif act == 'q':
            sys.exit()
        else:
            raise ValueError("Unknown action: {}".format(act))
    LOG_DIR = dirname
    from .fs import mkdir_p
    mkdir_p(dirname)
    _set_file(os.path.join(dirname, 'log.log'))


def auto_set_dir(action=None, name=None):
    """
    Use :func:`logger.set_logger_dir` to set log directory to
    "./train_log/{scriptname}:{name}". "scriptname" is the name of the main python file currently running"""
    mod = sys.modules['__main__']
    basename = os.path.basename(mod.__file__)
    auto_dirname = os.path.join('train_log', basename[:basename.rfind('.')])
    if name:
        auto_dirname += ':%s' % name
    set_logger_dir(auto_dirname, action=action)
