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

from .utils import memoized
from .fs import mkdir_p

__all__ = []

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


def get_time_str():
    return datetime.now().strftime('%m%d-%H%M%S')
# logger file and directory:
global LOG_FILE, LOG_DIR
LOG_DIR = None
def _set_file(path):
    if os.path.isfile(path):
        backup_name = path + '.' + get_time_str()
        shutil.move(path, backup_name)
        info("Log file '{}' backuped to '{}'".format(path, backup_name))
    hdl = logging.FileHandler(
        filename=path, encoding='utf-8', mode='w')
    hdl.setFormatter(_MyFormatter(datefmt='%m%d %H:%M:%S'))
    _logger.addHandler(hdl)

def set_logger_dir(dirname, action=None):
    """
    Set the directory for global logging.
    :param dirname: log directory
    :param action: an action (k/b/d/n) to be performed. Will ask user by default.
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
            info("Directory '{}' backuped to '{}'".format(dirname, backup_name))
        elif act == 'd':
            shutil.rmtree(dirname)
        elif act == 'n':
            dirname = dirname + get_time_str()
            info("Use a new log directory {}".format(dirname))
        elif act == 'k':
            pass
        else:
            raise ValueError("Unknown action: {}".format(act))
    LOG_DIR = dirname
    mkdir_p(dirname)
    LOG_FILE = os.path.join(dirname, 'log.log')
    _set_file(LOG_FILE)

# export logger functions
for func in ['info', 'warning', 'error', 'critical', 'warn', 'exception', 'debug']:
    locals()[func] = getattr(_logger, func)

def disable_logger():
    """ disable all logging ability from this moment"""
    for func in ['info', 'warning', 'error', 'critical', 'warn', 'exception', 'debug']:
        globals()[func] = lambda x: None

def auto_set_dir(action=None, overwrite_setting=False):
    """ set log directory to a subdir inside 'train_log', with the name being
    the main python file currently running"""
    if LOG_DIR is not None and not overwrite_setting:
        return
    mod = sys.modules['__main__']
    basename = os.path.basename(mod.__file__)
    set_logger_dir(
            os.path.join('train_log',
                basename[:basename.rfind('.')]),
            action=action)
