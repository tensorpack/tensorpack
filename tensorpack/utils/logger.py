# -*- coding: UTF-8 -*-
# File: logger.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import logging
import os, shutil
import os.path
from termcolor import colored
from datetime import datetime
from six.moves import input
import sys

from .fs import mkdir_p

__all__ = []

class MyFormatter(logging.Formatter):
    def format(self, record):
        date = colored('[%(asctime)s %(lineno)d@%(filename)s:%(name)s]', 'green')
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
        else:
            self._fmt = fmt
        return super(MyFormatter, self).format(record)

def getlogger():
    logger = logging.getLogger('tensorpack')
    logger.propagate = False
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(MyFormatter(datefmt='%d %H:%M:%S'))
    logger.addHandler(handler)
    return logger

def get_time_str():
    return datetime.now().strftime('%m%d-%H%M%S')

logger = getlogger()

# logger file and directory:
global LOG_FILE, LOG_DIR
def _set_file(path):
    if os.path.isfile(path):
        backup_name = path + '.' + get_time_str()
        shutil.move(path, backup_name)
        info("Log file '{}' backuped to '{}'".format(path, backup_name))
    hdl = logging.FileHandler(
        filename=path, encoding='utf-8', mode='w')
    hdl.setFormatter(MyFormatter(datefmt='%d %H:%M:%S'))
    logger.addHandler(hdl)

def set_logger_dir(dirname, action=None):
    """
    Set the directory for global logging.
    :param dirname: log directory
    :param action: an action (k/b/d/n) to be performed. Will ask user by default.
    """
    global LOG_FILE, LOG_DIR
    if os.path.isdir(dirname):
        logger.warn("""\
Directory {} exists! Please either backup/delete it, or use a new directory \
unless you're resuming from a previous task.""".format(dirname))
        logger.info("Select Action: k (keep) / b (backup) / d (delete) / n (new):")
        if not action:
            while True:
                act = input().lower().strip()
                if act:
                    break
        else:
            act = action
        if act == 'b':
            backup_name = dirname + get_time_str()
            shutil.move(dirname, backup_name)
            info("Directory'{}' backuped to '{}'".format(dirname, backup_name))
        elif act == 'd':
            shutil.rmtree(dirname)
        elif act == 'n':
            dirname = dirname + get_time_str()
            info("Use a different log directory {}".format(dirname))
        elif act == 'k':
            pass
        else:
            raise ValueError("Unknown action: {}".format(act))
    LOG_DIR = dirname
    mkdir_p(dirname)
    LOG_FILE = os.path.join(dirname, 'log.log')
    _set_file(LOG_FILE)

def disable_logger():
    for func in ['info', 'warning', 'error', 'critical', 'warn', 'exception', 'debug']:
        globals()[func] = lambda x: None

# export logger functions
for func in ['info', 'warning', 'error', 'critical', 'warn', 'exception', 'debug']:
    locals()[func] = getattr(logger, func)
