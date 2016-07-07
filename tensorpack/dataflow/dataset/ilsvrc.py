#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: ilsvrc.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>
import os
import tarfile
import cv2
import numpy as np
from six.moves import range

from ...utils import logger, get_rng, get_dataset_dir, memoized
from ...utils.timer import timed_operation
from ...utils.fs import mkdir_p, download
from ..base import DataFlow

__all__ = ['ILSVRCMeta', 'ILSVRC12']

try:
    import lmdb
except ImportError:
    logger.warn("Error in 'import lmdb'. ILSVRC12CaffeLMDB won't be available.")
else:
    __all__.append('ILSVRC12CaffeLMDB')

@memoized
def log_once(s): logger.warn(s)

CAFFE_ILSVRC12_URL = "http://dl.caffe.berkeleyvision.org/caffe_ilsvrc12.tar.gz"
CAFFE_PROTO_URL = "https://github.com/BVLC/caffe/raw/master/src/caffe/proto/caffe.proto"

"""
Preprocess training set like this:
    cd train
    for i in *.tar; do dir=${i%.tar}; echo $dir; mkdir -p $dir; tar xf $i -C $dir; done
"""

class ILSVRCMeta(object):
    """
    Provide metadata for ILSVRC dataset.
    """
    def __init__(self, dir=None):
        if dir is None:
            dir = get_dataset_dir('ilsvrc_metadata')
        self.dir = dir
        mkdir_p(self.dir)
        self.caffe_pb_file = os.path.join(self.dir, 'caffe_pb2.py')
        if not os.path.isfile(self.caffe_pb_file):
            self._download_caffe_meta()

    def get_synset_words_1000(self):
        """
        :returns a dict of {cls_number: cls_name}
        """
        fname = os.path.join(self.dir, 'synset_words.txt')
        assert os.path.isfile(fname)
        lines = [x.strip() for x in open(fname).readlines()]
        return dict(enumerate(lines))

    def _download_caffe_meta(self):
        fpath = download(CAFFE_ILSVRC12_URL, self.dir)
        tarfile.open(fpath, 'r:gz').extractall(self.dir)

        proto_path = download(CAFFE_PROTO_URL, self.dir)
        ret = os.system('cd {} && protoc caffe.proto --python_out .'.format(self.dir))
        assert ret == 0, \
                "caffe proto compilation failed! Did you install protoc?"

    def get_image_list(self, name):
        """
        :param name: 'train' or 'val' or 'test'
        :returns list of (image filename, cls)
        """
        assert name in ['train', 'val', 'test']
        fname = os.path.join(self.dir, name + '.txt')
        assert os.path.isfile(fname)
        with open(fname) as f:
            ret = []
            for line in f.readlines():
                name, cls = line.strip().split()
                ret.append((name, int(cls)))
        return ret

    def get_per_pixel_mean(self, size=None):
        """
        :param size: return image size in [h, w]. default to (256, 256)
        :returns per-pixel mean as an array of shape (h, w, 3) in range [0, 255]
        """
        import imp
        caffepb = imp.load_source('caffepb', self.caffe_pb_file)
        obj = caffepb.BlobProto()

        mean_file = os.path.join(self.dir, 'imagenet_mean.binaryproto')
        with open(mean_file) as f:
            obj.ParseFromString(f.read())
        arr = np.array(obj.data).reshape((3, 256, 256)).astype('float32')
        arr = np.transpose(arr, [1,2,0])
        if size is not None:
            arr = cv2.resize(arr, size[::-1])
        return arr

class ILSVRC12(DataFlow):
    def __init__(self, dir, name, meta_dir=None, shuffle=True):
        """
        name: 'train' or 'val' or 'test'
        """
        assert name in ['train', 'test', 'val']
        self.full_dir = os.path.join(dir, name)
        self.shuffle = shuffle
        self.meta = ILSVRCMeta(meta_dir)
        self.imglist = self.meta.get_image_list(name)
        self.rng = get_rng(self)

    def size(self):
        return len(self.imglist)

    def reset_state(self):
        """
        reset rng for shuffle
        """
        self.rng = get_rng(self)

    def get_data(self):
        """
        Produce original images or shape [h, w, 3], and label
        """
        idxs = np.arange(len(self.imglist))
        if self.shuffle:
            self.rng.shuffle(idxs)
        for k in idxs:
            tp = self.imglist[k]
            fname = os.path.join(self.full_dir, tp[0]).strip()
            im = cv2.imread(fname, cv2.IMREAD_COLOR)
            assert im is not None, fname
            if im.ndim == 2:
                im = np.expand_dims(im, 2).repeat(3,2)
            yield [im, tp[1]]

class ILSVRC12CaffeLMDB(DataFlow):
    def __init__(self, lmdb_dir, shuffle=True):
        """
        :param shuffle: about 3 times slower
        """
        self._lmdb = lmdb.open(lmdb_dir, readonly=True, lock=False,
                map_size=1099511627776 * 2, max_readers=100)
        self._txn = self._lmdb.begin()
        self._meta = ILSVRCMeta()
        self._shuffle = shuffle
        self.rng = get_rng(self)
        self._txn = self._lmdb.begin()
        self._size = self._txn.stat()['entries']
        if shuffle:
            with timed_operation("Loading LMDB keys ..."):
                self.keys = [k for k, _ in self._txn.cursor()]

    def reset_state(self):
        self._txn = self._lmdb.begin()
        self.rng = get_rng(self)

    def size(self):
        return self._size

    def get_data(self):
        import imp
        cpb = imp.load_source('cpb', self._meta.caffe_pb_file)
        datum = cpb.Datum()

        def parse(k, v):
            try:
                datum.ParseFromString(v)
                img = np.fromstring(datum.data, dtype=np.uint8)
                img = img.reshape(datum.channels, datum.height, datum.width)
            except Exception:
                log_once("Cannot read key {}".format(k))
                return None
            return [img.transpose(1, 2, 0), datum.label]

        if not self._shuffle:
            c = self._txn.cursor()
            while c.next():
                k, v = c.item()
                v = parse(k, v)
                if v: yield v
        else:
            s = self.size()
            for i in range(s):
                k = self.rng.choice(self.keys)
                v = self._txn.get(k)
                v = parse(k, v)
                if v: yield v

if __name__ == '__main__':
    meta = ILSVRCMeta()
    print(meta.get_per_pixel_mean())
    #print(meta.get_synset_words_1000())

    #ds = ILSVRC12('/home/wyx/data/imagenet', 'val')

    ds = ILSVRC12CaffeLMDB('/home/yuxinwu/', True)

    for k in ds.get_data():
        from IPython import embed; embed()
        break
