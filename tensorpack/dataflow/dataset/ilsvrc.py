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
from ...utils.loadcaffe import get_caffe_pb
from ...utils.fs import mkdir_p, download
from ..base import RNGDataFlow

__all__ = ['ILSVRCMeta', 'ILSVRC12']

@memoized
def log_once(s): logger.warn(s)

CAFFE_ILSVRC12_URL = "http://dl.caffe.berkeleyvision.org/caffe_ilsvrc12.tar.gz"

# TODO move caffe_pb outside
class ILSVRCMeta(object):
    """
    Some metadata for ILSVRC dataset.
    """
    def __init__(self, dir=None):
        if dir is None:
            dir = get_dataset_dir('ilsvrc_metadata')
        self.dir = dir
        mkdir_p(self.dir)
        self.caffepb = get_caffe_pb()
        f = os.path.join(self.dir, 'synsets.txt')
        if not os.path.isfile(f):
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

    def get_image_list(self, name):
        """
        :param name: 'train' or 'val' or 'test'
        :returns: list of (image filename, cls)
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
        :returns: per-pixel mean as an array of shape (h, w, 3) in range [0, 255]
        """
        obj = self.caffepb.BlobProto()

        mean_file = os.path.join(self.dir, 'imagenet_mean.binaryproto')
        with open(mean_file, 'rb') as f:
            obj.ParseFromString(f.read())
        arr = np.array(obj.data).reshape((3, 256, 256)).astype('float32')
        arr = np.transpose(arr, [1,2,0])
        if size is not None:
            arr = cv2.resize(arr, size[::-1])
        return arr

class ILSVRC12(RNGDataFlow):
    def __init__(self, dir, name, meta_dir=None, shuffle=True):
        """
        :param dir: A directory containing a subdir named `name`, where the
            original ILSVRC12_`name`.tar gets decompressed.
        :param name: 'train' or 'val' or 'test'

        Dir should have the following structure:

        .. code-block:: none

            dir/
              train/
                n02134418/
                  n02134418_198.JPEG
                  ...
                ...
              val/
                ILSVRC2012_val_00000001.JPEG
                ...
              test/
                ILSVRC2012_test_00000001.JPEG
                ...

        After decompress ILSVRC12_img_train.tar, you can use the following
        command to build the above structure for `train/`:

        .. code-block:: none

            find -type f | parallel -P 10 'mkdir -p {/.} && tar xf {} -C {/.}'

            Or:
            for i in *.tar; do dir=${i%.tar}; echo $dir; mkdir -p $dir; tar xf $i -C $dir; done
        """
        assert name in ['train', 'test', 'val']
        self.full_dir = os.path.join(dir, name)
        assert os.path.isdir(self.full_dir), self.full_dir
        self.shuffle = shuffle
        self.meta = ILSVRCMeta(meta_dir)
        self.imglist = self.meta.get_image_list(name)

    def size(self):
        return len(self.imglist)

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


if __name__ == '__main__':
    meta = ILSVRCMeta()
    print(meta.get_per_pixel_mean())
    #print(meta.get_synset_words_1000())

    #ds = ILSVRC12('/home/wyx/data/imagenet', 'val')

    for k in ds.get_data():
        from IPython import embed; embed()
        break
