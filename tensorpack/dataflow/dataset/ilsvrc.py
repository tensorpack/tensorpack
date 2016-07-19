#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: ilsvrc.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>
import os
import tarfile
import cv2
import numpy as np
from six.moves import range
import xml.etree.ElementTree as ET

from ...utils import logger, get_rng, get_dataset_path, memoized
from ...utils.loadcaffe import get_caffe_pb
from ...utils.fs import mkdir_p, download
from ...utils.timer import timed_operation
from ..base import RNGDataFlow

__all__ = ['ILSVRCMeta', 'ILSVRC12']

@memoized
def log_once(s): logger.warn(s)

CAFFE_ILSVRC12_URL = "http://dl.caffe.berkeleyvision.org/caffe_ilsvrc12.tar.gz"

class ILSVRCMeta(object):
    """
    Some metadata for ILSVRC dataset.
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
        :returns a dict of {cls_number: cls_name}
        """
        fname = os.path.join(self.dir, 'synset_words.txt')
        assert os.path.isfile(fname)
        lines = [x.strip() for x in open(fname).readlines()]
        return dict(enumerate(lines))

    def get_synset_1000(self):
        """
        :returns a dict of {cls_number: synset_id}
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
    def __init__(self, dir, name, meta_dir=None, shuffle=True,
            dir_structure='original', include_bb=False):
        """
        :param dir: A directory containing a subdir named `name`, where the
            original ILSVRC12_`name`.tar gets decompressed.
        :param name: 'train' or 'val' or 'test'
        :param dir_structure: The dir structure of 'val' or 'test'.
            If is 'original' then keep the original decompressed dir with list
            of image files (as below). If equals to 'train', use the `train/` dir
            structure with class name as subdirectories.
        :param include_bb: Include the bounding box. Useful in training.

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
              bbox/
                n02134418/
                  n02134418_198.xml
                  ...
                ...

        After decompress ILSVRC12_img_train.tar, you can use the following
        command to build the above structure for `train/`:

        .. code-block:: none
            tar xvf ILSVRC12_img_train.tar -C train && cd train
            find -type f -name '*.tar' | parallel -P 10 'echo {} && mkdir -p {/.} && tar xf {} -C {/.}'
            Or:
            for i in *.tar; do dir=${i%.tar}; echo $dir; mkdir -p $dir; tar xf $i -C $dir; done

        """
        assert name in ['train', 'test', 'val']
        self.full_dir = os.path.join(dir, name)
        self.name = name
        assert os.path.isdir(self.full_dir), self.full_dir
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
        """
        Produce original images of shape [h, w, 3], and label,
        and optionally a bbox of [xmin, ymin, xmax, ymax]
        """
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
                im = np.expand_dims(im, 2).repeat(3,2)
            if self.include_bb:
                bb = self.bblist[k]
                if bb is None:
                    bb = [0, 0, im.shape[1]-1, im.shape[0]-1]
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
            #box[0] /= size[0]
            #box[1] /= size[1]
            #box[2] /= size[0]
            #box[3] /= size[1]
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
    #print(meta.get_synset_words_1000())

    ds = ILSVRC12('/home/wyx/data/fake_ilsvrc/', 'train', include_bb=True,
            shuffle=False)
    ds.reset_state()

    for k in ds.get_data():
        from IPython import embed; embed()
        break
