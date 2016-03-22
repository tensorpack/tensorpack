#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: ilsvrc.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>
import os
import tarfile
from ...utils.fs import mkdir_p, download

__all__ = ['ILSVRCMeta']

CAFFE_URL = "http://dl.caffe.berkeleyvision.org/caffe_ilsvrc12.tar.gz"

class ILSVRCMeta(object):
    def __init__(self, dir=None):
        if dir is None:
            dir = os.path.join(os.path.dirname(__file__), 'ilsvrc_metadata')
        self.dir = dir
        mkdir_p(self.dir)

    def get_synset_words_1000(self):
        fname = os.path.join(self.dir, 'synset_words.txt')
        if not os.path.isfile(fname):
            self.download_caffe_meta()
        assert os.path.isfile(fname)
        lines = [x.strip() for x in open(fname).readlines()]
        return dict(enumerate(lines))

    def download_caffe_meta(self):
        fpath = download(CAFFE_URL, self.dir)
        tarfile.open(fpath, 'r:gz').extractall(self.dir)

if __name__ == '__main__':
    meta = ILSVRCMeta()
    print meta.get_synset_words_1000()
