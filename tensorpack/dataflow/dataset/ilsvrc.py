#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: ilsvrc.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>
import os
import tarfile
import cv2
import six
import numpy as np
import xml.etree.ElementTree as ET

from ...utils import logger
from ...utils.loadcaffe import get_caffe_pb
from ...utils.fs import mkdir_p, download, get_dataset_path
from ...utils.timer import timed_operation
from ..base import RNGDataFlow

__all__ = ['ILSVRCMeta', 'ILSVRC12']

CAFFE_ILSVRC12_URL = "http://dl.caffe.berkeleyvision.org/caffe_ilsvrc12.tar.gz"


class ILSVRCMeta(object):
    """
    Provide methods to access metadata for ILSVRC dataset.
    """

    def __init__(self, dir=None):
        if dir is None:
            dir = get_dataset_path('ilsvrc_metadata')
        self.dir = dir
        mkdir_p(self.dir)
        self.caffepb = get_caffe_pb()
        f = os.path.join(self.dir, 'synsets.txt')
        if not os.path.isfile(f):
            self._download_caffe_meta()

    def get_synset_words_1000(self):
        """
        Returns:
            dict: {cls_number: cls_name}
        """
        fname = os.path.join(self.dir, 'synset_words.txt')
        assert os.path.isfile(fname)
        lines = [x.strip() for x in open(fname).readlines()]
        return dict(enumerate(lines))

    def get_synset_1000(self):
        """
        Returns:
            dict: {cls_number: synset_id}
        """
        fname = os.path.join(self.dir, 'synsets.txt')
        assert os.path.isfile(fname)
        lines = [x.strip() for x in open(fname).readlines()]
        return dict(enumerate(lines))

    def _download_caffe_meta(self):
        fpath = download(CAFFE_ILSVRC12_URL, self.dir)
        tarfile.open(fpath, 'r:gz').extractall(self.dir)

    def get_image_list(self, name):
        """
        Args:
            name (str): 'train' or 'val' or 'test'
        Returns:
            list: list of (image filename, label)
        """
        assert name in ['train', 'val', 'test']
        fname = os.path.join(self.dir, name + '.txt')
        assert os.path.isfile(fname)
        with open(fname) as f:
            ret = []
            for line in f.readlines():
                name, cls = line.strip().split()
                ret.append((name, int(cls)))
        assert len(ret)
        return ret

    def get_per_pixel_mean(self, size=None):
        """
        Args:
            size (tuple): image size in (h, w). Defaults to (256, 256).
        Returns:
            np.ndarray: per-pixel mean of shape (h, w, 3 (BGR)) in range [0, 255].
        """
        obj = self.caffepb.BlobProto()

        mean_file = os.path.join(self.dir, 'imagenet_mean.binaryproto')
        with open(mean_file, 'rb') as f:
            obj.ParseFromString(f.read())
        arr = np.array(obj.data).reshape((3, 256, 256)).astype('float32')
        arr = np.transpose(arr, [1, 2, 0])
        if size is not None:
            arr = cv2.resize(arr, size[::-1])
        return arr


class ILSVRC12(RNGDataFlow):
    """
    Produces uint8 ILSVRC12 images of shape [h, w, 3(BGR)], and a label between [0, 999],
    and optionally a bounding box of [xmin, ymin, xmax, ymax].
    """
    def __init__(self, dir, name, meta_dir=None, shuffle=None,
                 dir_structure='original', include_bb=False):
        """
        Args:
            dir (str): A directory containing a subdir named ``name``, where the
                original ``ILSVRC12_img_{name}.tar`` gets decompressed.
            name (str): 'train' or 'val' or 'test'.
            shuffle (bool): shuffle the dataset.
                Defaults to True if name=='train'.
            dir_structure (str): The dir structure of 'val' and 'test' directory.
                If is 'original', it expects the original decompressed
                directory, which only has list of image files (as below).
                If set to 'train', it expects the same two-level
                directory structure simlar to 'train/'.
            include_bb (bool): Include the bounding box. Maybe useful in training.

        Examples:

        When `dir_structure=='original'`, `dir` should have the following structure:

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

        With ILSVRC12_img_*.tar, you can use the following
        command to build the above structure:

        .. code-block:: none

            mkdir val && tar xvf ILSVRC12_img_val.tar -C val
            mkdir test && tar xvf ILSVRC12_img_test.tar -C test
            mkdir train && tar xvf ILSVRC12_img_train.tar -C train && cd train
            find -type f -name '*.tar' | parallel -P 10 'echo {} && mkdir -p {/.} && tar xf {} -C {/.}'
        """
        assert name in ['train', 'test', 'val'], name
        assert os.path.isdir(dir), dir
        self.full_dir = os.path.join(dir, name)
        self.name = name
        assert os.path.isdir(self.full_dir), self.full_dir
        if shuffle is None:
            shuffle = name == 'train'
        self.shuffle = shuffle
        meta = ILSVRCMeta(meta_dir)
        self.imglist = meta.get_image_list(name)
        self.dir_structure = dir_structure
        self.synset = meta.get_synset_1000()

        if include_bb:
            bbdir = os.path.join(dir, 'bbox') if not \
                isinstance(include_bb, six.string_types) else include_bb
            assert name == 'train', 'Bounding box only available for training'
            self.bblist = ILSVRC12.get_training_bbox(bbdir, self.imglist)
        self.include_bb = include_bb

    def size(self):
        return len(self.imglist)

    def get_data(self):
        idxs = np.arange(len(self.imglist))
        add_label_to_fname = (self.name != 'train' and self.dir_structure != 'original')
        if self.shuffle:
            self.rng.shuffle(idxs)
        for k in idxs:
            fname, label = self.imglist[k]
            if add_label_to_fname:
                fname = os.path.join(self.full_dir, self.synset[label], fname)
            else:
                fname = os.path.join(self.full_dir, fname)
            im = cv2.imread(fname.strip(), cv2.IMREAD_COLOR)
            assert im is not None, fname
            if im.ndim == 2:
                im = np.expand_dims(im, 2).repeat(3, 2)
            if self.include_bb:
                bb = self.bblist[k]
                if bb is None:
                    bb = [0, 0, im.shape[1] - 1, im.shape[0] - 1]
                yield [im, label, bb]
            else:
                yield [im, label]

    @staticmethod
    def get_training_bbox(bbox_dir, imglist):
        ret = []

        def parse_bbox(fname):
            root = ET.parse(fname).getroot()
            size = root.find('size').getchildren()
            size = map(int, [size[0].text, size[1].text])

            box = root.find('object').find('bndbox').getchildren()
            box = map(lambda x: float(x.text), box)
            # box[0] /= size[0]
            # box[1] /= size[1]
            # box[2] /= size[0]
            # box[3] /= size[1]
            return np.asarray(box, dtype='float32')

        with timed_operation('Loading Bounding Boxes ...'):
            cnt = 0
            import tqdm
            for k in tqdm.trange(len(imglist)):
                fname = imglist[k][0]
                fname = fname[:-4] + 'xml'
                fname = os.path.join(bbox_dir, fname)
                try:
                    ret.append(parse_bbox(fname))
                    cnt += 1
                except KeyboardInterrupt:
                    raise
                except:
                    ret.append(None)
            logger.info("{}/{} images have bounding box.".format(cnt, len(imglist)))
        return ret


if __name__ == '__main__':
    meta = ILSVRCMeta()
    # print(meta.get_synset_words_1000())

    ds = ILSVRC12('/home/wyx/data/fake_ilsvrc/', 'train', include_bb=True,
                  shuffle=False)
    ds.reset_state()

    for k in ds.get_data():
        from IPython import embed
        embed()
        break
