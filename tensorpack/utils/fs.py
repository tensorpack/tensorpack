#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: fs.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import os
import sys
from six.moves import urllib
import errno
from . import logger
from .utils import execute_only_once

__all__ = ['mkdir_p', 'download', 'recursive_walk', 'get_dataset_path']


def mkdir_p(dirname):
    """ Make a dir recursively, but do nothing if the dir exists

    Args:
        dirname(str):
    """
    assert dirname is not None
    if dirname == '' or os.path.isdir(dirname):
        return
    try:
        os.makedirs(dirname)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise e


def download(url, dir):
    """
    Download URL to a directory. Will figure out the filename automatically
    from URL.
    """
    mkdir_p(dir)
    fname = url.split('/')[-1]
    fpath = os.path.join(dir, fname)

    def _progress(count, block_size, total_size):
        sys.stdout.write('\r>> Downloading %s %.1f%%' %
                         (fname,
                             min(float(count * block_size) / total_size,
                                 1.0) * 100.0))
        sys.stdout.flush()
    try:
        fpath, _ = urllib.request.urlretrieve(url, fpath, reporthook=_progress)
        statinfo = os.stat(fpath)
        size = statinfo.st_size
    except:
        logger.error("Failed to download {}".format(url))
        raise
    assert size > 0, "Download an empty file!"
    sys.stdout.write('\n')
    # TODO human-readable size
    print('Succesfully downloaded ' + fname + " " + str(size) + ' bytes.')
    return fpath


def recursive_walk(rootdir):
    """
    Yields:
        str: All files in rootdir, recursively.
    """
    for r, dirs, files in os.walk(rootdir):
        for f in files:
            yield os.path.join(r, f)


def get_dataset_path(*args):
    """
    Get the path to some dataset under ``$TENSORPACK_DATASET``.

    Args:
        args: strings to be joined to form path.

    Returns:
        str: path to the dataset.
    """
    d = os.environ.get('TENSORPACK_DATASET', None)
    if d is None:
        old_d = os.path.abspath(os.path.join(
            os.path.dirname(__file__), '..', 'dataflow', 'dataset'))
        old_d_ret = os.path.join(d, *args)
        new_d = os.path.expanduser('~/tensorpack_data')
        if os.path.isdir(old_d_ret):
            # there is an old dir containing data, use it for back-compat
            logger.warn("You seem to have old data at {}. This is no longer \
                the default location. You'll need to move it to {} \
                (the new default location) or another directory set by \
                $TENSORPACK_DATASET.".format(old_d, new_d))
        d = new_d
        if execute_only_once():
            logger.warn("$TENSORPACK_DATASET not set, using {} for dataset.".format(d))
        if not os.path.isdir(d):
            mkdir_p(d)
            logger.info("Created the directory {}.".format(d))
    assert os.path.isdir(d), d
    return os.path.join(d, *args)


if __name__ == '__main__':
    download('http://dl.caffe.berkeleyvision.org/caffe_ilsvrc12.tar.gz', '.')
