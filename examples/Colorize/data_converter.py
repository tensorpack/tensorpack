#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Patrick Wieschollek <mail@patwie.com>, 2016 University of Tuebingen


import tarfile
import argparse
import cv2
import os
import numpy as np
from tensorpack import *

"""
Just convert Place2-dataset (http://places2.csail.mit.edu) to an LMDB file for
more efficient reading.


example:

    python place2_lmdb_generator.py --tar train_large_places365standard.tar \
                                    --lmdb /data/train_large_places365standard.lmdb \
                                    --labels places365_train_standard.txt

    or

    python place2_lmdb_generator.py --tar val_large.tar' \
                                    --lmdb /data/val_large.lmdb \
                                    --labels places365_val.txt
"""


class Place2Reader(RNGDataFlow):
    """Read images directly from tar file without unpacking
    """
    def __init__(self, tar, labels):
        super(Place2Reader, self).__init__()
        assert os.path.isfile(tar)
        assert os.path.isfile(labels)
        self.tar = tarfile.open(tar)
        self.tar_name = tar
        self.labels = dict()
        with open(labels, 'r') as f:
            for line in f.readlines():
                path, clazz = line.split(' ')
                clazz = int(clazz)
                self.labels[path] = clazz

    def get_data(self):
        if 'train' in self.tar_name:
            for member in self.tar:
                cur_name = member.name.replace('data_large', '')
                f = self.tar.extractfile(member)
                jpeg = np.asarray(bytearray(f.read()), dtype=np.uint8)
                f.close()
                yield [jpeg, self.labels[cur_name]]
        else:
            imglist = self.tar.getnames()
            for member in imglist:
                cur_name = member.replace('val_large/', '')
                if '.jpg' in member:
                    try:
                        f = self.tar.extractfile(member)
                        jpeg = np.asarray(bytearray(f.read()), dtype=np.uint8)
                        f.close()
                        yield [jpeg, self.labels[cur_name]]
                    except Exception:
                        print("skip %s" % cur_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tar', help='path to tar file',
                        default='/graphics/projects/scratch/datasets/places2/untouched/val_large.tar')
    parser.add_argument('--labels', help='path to label txt file',
                        default='/graphics/projects/scratch/datasets/places2/untouched/places365_val.txt')
    parser.add_argument('--lmdb', help='path to database (to be written)',
                        default='/graphics/projects/scratch/datasets/places2/val_large.lmdb')
    parser.add_argument('--debug', action='store_true',
                        help='just show the images')
    args = parser.parse_args()

    if args.debug:
        ds = Place2Reader(args.tar, args.labels)
        ds.reset_state()
        for jpeg, label in ds.get_data():
            rgb = cv2.imdecode(np.asarray(jpeg), cv2.IMREAD_COLOR)
            cv2.imshow("RGB image from Place2-dataset", rgb)
            print("label %i" % label)
            cv2.waitKey(0)
    else:
        ds = Place2Reader(args.tar, args.labels)
        ds.reset_state()
        dftools.dump_dataflow_to_lmdb(ds, args.lmdb)
