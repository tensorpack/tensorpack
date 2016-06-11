#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: bsds500.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import os, glob
import cv2
import numpy as np
from scipy.io import loadmat
from ...utils import logger, get_rng
from ...utils.fs import download
from ..base import DataFlow
from .common import get_dataset_dir

__all__ = ['BSDS500']


DATA_URL = "http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/BSR/BSR_bsds500.tgz"
IMG_W, IMG_H = 481, 321

class BSDS500(DataFlow):
    """
    `Berkeley Segmentation Data Set and Benchmarks 500
    <http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html#bsds500>`_.

    Produce (image, label) pair, where image has shape (321, 481, 3) and
    ranges in [0,255]. Label is binary and has shape (321, 481).
    Those pixels annotated as boundaries by >= 3 annotators are
    considered positive examples. This is used in `Holistically-Nested Edge Detection
    <http://arxiv.org/abs/1504.06375>`_.
    """

    def __init__(self, name, data_dir=None, shuffle=True):
        """
        :param name: 'train', 'test', 'val'
        :param data_dir: a directory containing the original 'BSR' directory.
        """
        # check and download data
        if data_dir is None:
            data_dir = get_dataset_dir('bsds500_data')
        if not os.path.isdir(os.path.join(data_dir, 'BSR')):
            download(DATA_URL, data_dir)
            filename = DATA_URL.split('/')[-1]
            filepath = os.path.join(data_dir, filename)
            import tarfile
            tarfile.open(filepath, 'r:gz').extractall(data_dir)
        self.data_root = os.path.join(data_dir, 'BSR', 'BSDS500', 'data')
        assert os.path.isdir(self.data_root)

        self.shuffle = shuffle
        assert name in ['train', 'test', 'val']
        self._load(name)
        self.rng = get_rng(self)

    def reset_state(self):
        self.rng = get_rng(self)

    def _load(self, name):
        image_glob = os.path.join(self.data_root, 'images', name, '*.jpg')
        image_files = glob.glob(image_glob)
        gt_dir = os.path.join(self.data_root, 'groundTruth', name)
        self.data = np.zeros((len(image_files), IMG_H, IMG_W, 3), dtype='uint8')
        self.label = np.zeros((len(image_files), IMG_H, IMG_W), dtype='bool')

        for idx, f in enumerate(image_files):
            im = cv2.imread(f, cv2.IMREAD_COLOR)
            assert im is not None
            if im.shape[0] > im.shape[1]:
                im = np.transpose(im, (1,0,2))
            assert im.shape[:2] == (IMG_H, IMG_W), "{} != {}".format(im.shape[:2], (IMG_H, IMG_W))

            imgid = os.path.basename(f).split('.')[0]
            gt_file = os.path.join(gt_dir, imgid)
            gt = loadmat(gt_file)['groundTruth'][0]
            n_annot = gt.shape[0]
            gt = sum(gt[k]['Boundaries'][0][0] for k in range(n_annot))
            gt[gt < 3] = 0
            gt[gt != 0] = 1
            if gt.shape[0] > gt.shape[1]:
                gt = gt.transpose()
            assert gt.shape == (IMG_H, IMG_W)

            self.data[idx] = im
            self.label[idx] = gt

    def size(self):
        return self.data.shape[0]

    def get_data(self):
        idxs = np.arange(self.data.shape[0])
        if self.shuffle:
            self.rng.shuffle(idxs)
        for k in idxs:
            yield [self.data[k], self.label[k]]


if __name__ == '__main__':
    a = BSDS500('val')
    for k in a.get_data():
        cv2.imshow("haha", k[1].astype('uint8')*255)
        cv2.waitKey(1000)
