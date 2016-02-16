#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: logger.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import logging
import os, shutil
import os.path
from termcolor import colored
import sys
if not sys.version_info >= (3, 0):
    input = raw_input   # for compatibility

from .utils import mkdir_p

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

logger = getlogger()

# logger file and directory:
global LOG_FILE, LOG_DIR
def _set_file(path):
    if os.path.isfile(path):
        from datetime import datetime
        backup_name = path + datetime.now().strftime('.%d-%H%M%S')
        shutil.move(path, backup_name)
        info("Log file '{}' backuped to '{}'".format(path, backup_name))
    hdl = logging.FileHandler(
        filename=path, encoding='utf-8', mode='w')
    logger.addHandler(hdl)

def set_logger_dir(dirname):
    global LOG_FILE, LOG_DIR
    LOG_DIR = dirname
    if os.path.isdir(dirname):
        logger.info("Directory {} exists. Please either backup or delete it unless you're continue from a paused task." )
        logger.info("Select Action: k (keep) / b (backup) / d (delete):")
        act = input().lower()
        if act == 'b':
            from datetime import datetime
            backup_name = dirname + datetime.now().strftime('.%d-%H%M%S')
            shutil.move(dirname, backup_name)
            info("Log directory'{}' backuped to '{}'".format(dirname, backup_name))
        elif act == 'd':
            shutil.rmtree(dirname)
        elif act == 'k':
            pass
        else:
            raise ValueError("Unknown action: {}".format(act))
    mkdir_p(dirname)
    LOG_FILE = os.path.join(dirname, 'log.log')
    _set_file(LOG_FILE)


# export logger functions
for func in ['info', 'warning', 'error', 'critical', 'warn', 'exception', 'debug']:
    locals()[func] = getattr(logger, func)

# a SummaryWriter
writer = None

# a StatHolder
stat_holder = None
