# -*- coding: utf-8 -*-
# File: logger.py

"""
The logger module itself has the common logging functions of Python's
:class:`logging.Logger`. For example:

.. code-block:: python

    from tensorpack.utils import logger
    logger.set_logger_dir('train_log/test')
    logger.info("Test")
    logger.error("Error happened!")
"""


import logging
import os
import os.path
import shutil
import sys
from datetime import datetime
from six.moves import input
from termcolor import colored

__all__ = ['set_logger_dir', 'auto_set_dir', 'get_logger_dir']


class _MyFormatter(logging.Formatter):
    def format(self, record):
        date = colored('[%(asctime)s @%(filename)s:%(lineno)d]', 'green')
        msg = '%(message)s'
        if record.levelno == logging.WARNING:
            fmt = date + ' ' + colored('WRN', 'red', attrs=['blink']) + ' ' + msg
        elif record.levelno == logging.ERROR or record.levelno == logging.CRITICAL:
            fmt = date + ' ' + colored('ERR', 'red', attrs=['blink', 'underline']) + ' ' + msg
        elif record.levelno == logging.DEBUG:
            fmt = date + ' ' + colored('DBG', 'yellow', attrs=['blink']) + ' ' + msg
        else:
            fmt = date + ' ' + msg
        if hasattr(self, '_style'):
            # Python3 compatibility
            self._style._fmt = fmt
        self._fmt = fmt
        return super(_MyFormatter, self).format(record)


def _getlogger():
    # this file is synced to "dataflow" package as well
    package_name = "dataflow" if __name__.startswith("dataflow") else "tensorpack"
    logger = logging.getLogger(package_name)
    logger.propagate = False
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_MyFormatter(datefmt='%m%d %H:%M:%S'))
    logger.addHandler(handler)
    return logger


_logger = _getlogger()
_LOGGING_METHOD = ['info', 'warning', 'error', 'critical', 'exception', 'debug', 'setLevel', 'addFilter']
# export logger functions
for func in _LOGGING_METHOD:
    locals()[func] = getattr(_logger, func)
    __all__.append(func)
# 'warn' is deprecated in logging module
warn = _logger.warning
__all__.append('warn')


def _get_time_str():
    return datetime.now().strftime('%m%d-%H%M%S')


# globals: logger file and directory:
LOG_DIR = None
_FILE_HANDLER = None


def _set_file(path):
    global _FILE_HANDLER
    if os.path.isfile(path):
        backup_name = path + '.' + _get_time_str()
        shutil.move(path, backup_name)
        _logger.info("Existing log file '{}' backuped to '{}'".format(path, backup_name))  # noqa: F821
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
        action(str): an action of ["k","d","q"] to be performed
            when the directory exists. Will ask user by default.

                "d": delete the directory. Note that the deletion may fail when
                the directory is used by tensorboard.

                "k": keep the directory. This is useful when you resume from a
                previous training and want the directory to look as if the
                training was not interrupted.
                Note that this option does not load old models or any other
                old states for you. It simply does nothing.

    """
    dirname = os.path.normpath(dirname)
    global LOG_DIR, _FILE_HANDLER
    if _FILE_HANDLER:
        # unload and close the old file handler, so that we may safely delete the logger directory
        _logger.removeHandler(_FILE_HANDLER)
        del _FILE_HANDLER

    def dir_nonempty(dirname):
        # If directory exists and nonempty (ignore hidden files), prompt for action
        return os.path.isdir(dirname) and len([x for x in os.listdir(dirname) if x[0] != '.'])

    if dir_nonempty(dirname):
        if not action:
            _logger.warning("""\
Log directory {} exists! Use 'd' to delete it. """.format(dirname))
            _logger.warning("""\
If you're resuming from a previous run, you can choose to keep it.
Press any other key to exit. """)
        while not action:
            action = input("Select Action: k (keep) / d (delete) / q (quit):").lower().strip()
        act = action
        if act == 'b':
            backup_name = dirname + _get_time_str()
            shutil.move(dirname, backup_name)
            info("Directory '{}' backuped to '{}'".format(dirname, backup_name))  # noqa: F821
        elif act == 'd':
            shutil.rmtree(dirname, ignore_errors=True)
            if dir_nonempty(dirname):
                shutil.rmtree(dirname, ignore_errors=False)
        elif act == 'n':
            dirname = dirname + _get_time_str()
            info("Use a new log directory {}".format(dirname))  # noqa: F821
        elif act == 'k':
            pass
        else:
            raise OSError("Directory {} exits!".format(dirname))
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
        auto_dirname += '_%s' % name if os.name == 'nt' else ':%s' % name
    set_logger_dir(auto_dirname, action=action)


def get_logger_dir():
    """
    Returns:
        The logger directory, or None if not set.
        The directory is used for general logging, tensorboard events, checkpoints, etc.
    """
    return LOG_DIR
