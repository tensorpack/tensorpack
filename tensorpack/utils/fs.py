#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: fs.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import os
import sys
from six.moves import urllib
import errno
from . import logger

__all__ = ['mkdir_p', 'download', 'recursive_walk']


def mkdir_p(dirname):
    """ make a dir recursively, but do nothing if the dir exists"""
    assert dirname is not None
    if dirname == '' or os.path.isdir(dirname):
        return
    try:
        os.makedirs(dirname)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise e


def download(url, dir):
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
    for r, dirs, files in os.walk(rootdir):
        for f in files:
            yield os.path.join(r, f)

if __name__ == '__main__':
    download('http://dl.caffe.berkeleyvision.org/caffe_ilsvrc12.tar.gz', '.')
