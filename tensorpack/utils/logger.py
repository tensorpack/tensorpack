#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: logger.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import logging
import os
import os.path
from termcolor import colored
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

for func in ['info', 'warning', 'error', 'critical', 'warn', 'exception', 'debug']:
    locals()[func] = getattr(logger, func)

def set_file(path):
    if os.path.isfile(path):
        from datetime import datetime
        backup_name = path + datetime.now().strftime('.%d-%H%M%S')
        import shutil
        shutil.move(path, backup_name)
        info("Log file '{}' backuped to '{}'".format(path, backup_name))
    dirname = os.path.dirname(path)
    if not os.path.isdir(dirname):
        os.makedirs(dirname)
    hdl = logging.FileHandler(
        filename=path, encoding='utf-8', mode='w')
    logger.addHandler(hdl)

global LOG_DIR
LOG_DIR = "train_log"
def set_logger_dir(dirname):
    global LOG_DIR
    LOG_DIR = dirname
    mkdir_p(LOG_DIR)
    set_file(os.path.join(LOG_DIR, 'training.log'))

