# Author: Tao Hu <taohu620@gmail.com>


import os
import gzip
import numpy as np
import cv2

from ...utils import logger
from ..base import RNGDataFlow

__all__ = ['PascalVOC12']


class PascalVOC12(RNGDataFlow):
    def __init__(self, dir, meta_dir, name,
                 shuffle=None, dir_structure=None):

        assert name in ['train', 'val'], name
        assert os.path.isdir(dir), dir
        self.reset_state()
        self.dir = dir
        self.name = name

        if shuffle is None:
            shuffle = name == 'train'
        self.shuffle = shuffle
        self.imglist = []

        if name == 'train':
            f = open(os.path.join(meta_dir,"train_aug.txt"),"r")
        else:
            f = open(os.path.join(meta_dir, "val.txt"), "r")

        for line in f.readlines():
            self.imglist.append(line.strip("\n").split(" "))
        f.close()

        #self.imglist = self.imglist[:40]

    def size(self):
        return len(self.imglist)

    def get_data(self):
        idxs = np.arange(len(self.imglist))
        if self.shuffle:
            self.rng.shuffle(idxs)
        for k in idxs:
            fname, flabel = self.imglist[k]
            fname = os.path.join(self.dir, fname)
            flabel = os.path.join(self.dir,flabel)
            fname = cv2.imread(fname, cv2.IMREAD_COLOR)
            flabel = cv2.imread(flabel, cv2.IMREAD_GRAYSCALE)
            yield [fname, flabel]



if __name__ == '__main__':
    pass