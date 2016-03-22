#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: fs.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import os, sys
from six.moves import urllib

def mkdir_p(dirname):
    assert dirname is not None
    if dirname == '':
        return
    try:
        os.makedirs(dirname)
    except OSError as e:
        if e.errno != 17:
            raise e


def download(url, dir):
    mkdir_p(dir)
    fname = url.split('/')[-1]
    fpath = os.path.join(dir, fname)

    def _progress(count, block_size, total_size):
        sys.stdout.write('\r>> Downloading %s %.1f%%' %
                         (fname, float(count * block_size) / float(total_size) * 100.0))
        sys.stdout.flush()
    fpath, _ = urllib.request.urlretrieve(url, fpath, reporthook=_progress)
    statinfo = os.stat(fpath)
    sys.stdout.write('\n')
    print('Succesfully downloaded ' + fname + " " + str(statinfo.st_size) + ' bytes.')
    return fpath

if __name__ == '__main__':
    download('http://dl.caffe.berkeleyvision.org/caffe_ilsvrc12.tar.gz', '.')
