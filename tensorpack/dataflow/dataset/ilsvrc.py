#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: ilsvrc.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>
import os
import tarfile
import numpy as np
import scipy.ndimage as scimg

from ...utils import logger, get_rng
from ..base import DataFlow
from ...utils.fs import mkdir_p, download

__all__ = ['ILSVRCMeta']

CAFFE_ILSVRC12_URL = "http://dl.caffe.berkeleyvision.org/caffe_ilsvrc12.tar.gz"
CAFFE_PROTO_URL = "https://github.com/BVLC/caffe/raw/master/src/caffe/proto/caffe.proto"

class ILSVRCMeta(object):
    def __init__(self, dir=None):
        if dir is None:
            dir = os.path.join(os.path.dirname(__file__), 'ilsvrc_metadata')
        self.dir = dir
        mkdir_p(self.dir)
        self.caffe_pb_file = os.path.join(self.dir, 'caffe_pb2.py')
        if not os.path.isfile(self.caffe_pb_file):
            self.download_caffe_meta()

    def get_synset_words_1000(self):
        fname = os.path.join(self.dir, 'synset_words.txt')
        assert os.path.isfile(fname)
        lines = [x.strip() for x in open(fname).readlines()]
        return dict(enumerate(lines))

    def download_caffe_meta(self):
        fpath = download(CAFFE_ILSVRC12_URL, self.dir)
        tarfile.open(fpath, 'r:gz').extractall(self.dir)

        proto_path = download(CAFFE_PROTO_URL, self.dir)
        ret = os.system('cd {} && protoc caffe.proto --python_out .'.format(self.dir))
        assert ret == 0, "caffe proto compilation failed!"

    def get_image_list(self, name):
        assert name in ['train', 'val', 'test']
        fname = os.path.join(self.dir, name + '.txt')
        assert os.path.isfile(fname)
        with open(fname) as f:
            ret = []
            for line in f.readlines():
                name, cls = line.strip().split()
                ret.append((name, int(cls)))
            return ret

    def load_mean(self):
        """
        return per-pixel mean as an array of shape
         (3, 256, 256) in range [0, 255]
        """
        import imp
        caffepb = imp.load_source('caffepb', self.caffe_pb_file)
        obj = caffepb.BlobProto()

        mean_file = os.path.join(self.dir, 'imagenet_mean.binaryproto')
        with open(mean_file) as f:
            obj.ParseFromString(f.read())
        arr = np.array(obj.data).reshape((3, 256, 256))
        return arr

class ILSVRC12(DataFlow):
    def __init__(self, dir, name, meta_dir=None, shuffle=True):
        """
        name: 'train' or 'val' or 'test'
        """
        assert name in ['train', 'test', 'val']
        self.dir = dir
        self.name = name
        self.shuffle = shuffle
        self.meta = ILSVRCMeta(meta_dir)
        self.imglist = self.meta.get_image_list(name)
        self.rng = get_rng(self)

    def size(self):
        return len(self.imglist)

    def reset_state(self):
        self.rng = get_rng(self)

    def get_data(self):
        idxs = np.arange(len(self.imglist))
        if self.shuffle:
            self.rng.shuffle(idxs)
        for k in idxs:
            tp = self.imglist[k]
            fname = os.path.join(self.dir, self.name, tp[0])
            im = scimg.imread(fname)
            if len(im.shape) == 2:
                im = np.expand_dims(im, 2).repeat(3,2)
            yield [im, tp[1]]

if __name__ == '__main__':
    meta = ILSVRCMeta()
    print meta.load_mean()
    #print(meta.get_synset_words_1000())

    #ds = ILSVRC12('/home/wyx/data/imagenet', 'val')
    #for k in ds.get_data():
        #from IPython import embed; embed()
