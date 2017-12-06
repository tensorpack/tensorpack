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
Just convert Place-dataset (http://places.csail.mit.edu) to an LMDB file for
more efficient reading.

gunzip imagesPlaces205_resize.tar.gz


example:

    python place_lmdb_generator.py --tar imagesPlaces205_resize.tar \
                                   --lmdb /data/train_places205.lmdb \
                                   --labels trainvalsplit_places205/train_places205.csv
"""


class PlaceReader(RNGDataFlow):
    """Read images directly from tar file without unpacking.
    """
    def __init__(self, tar, labels):
        super(PlaceReader, self).__init__()
        assert os.path.isfile(tar)
        assert os.path.isfile(labels)
        self.tar = tarfile.open(tar)
        self.tar_name = tar
        self.labels_name = labels
        self.labels = dict()
        with open(labels, 'r') as f:
            for line in f.readlines():
                path, clazz = line.split(' ')
                clazz = int(clazz)
                self.labels[path] = clazz

    def get_data(self):
        for member in self.tar:
            f = self.tar.extractfile(member)
            lut = member.name.replace('data/vision/torralba/deeplearning/images256/', '')
            jpeg = np.asarray(bytearray(f.read()), dtype=np.uint8)
            f.close()
            try:
                yield [jpeg, self.labels[lut]]
            except Exception:
                pass  # not in training set


class ImageDecode(MapDataComponent):
    """Decode JPEG buffer to uint8 image array
    """

    def __init__(self, ds, mode='.jpg', dtype=np.uint8, index=0):
        def func(im_data):
            img = cv2.imdecode(np.asarray(
                bytearray(im_data), dtype=dtype), cv2.IMREAD_COLOR)
            return img[:, :, ::-1]
        super(ImageDecode, self).__init__(ds, func, index=index)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tar', help='path to tar file',
                        default='/dataset/MIT_places/imagesPlaces205_resize.tar')
    parser.add_argument('--labels', help='path to label txt file',
                        default='/dataset/MIT_places/trainvalsplit_places205/val_places205.csv')
    parser.add_argument('--lmdb', help='path to database (to be written)',
                        default='/dataset/MIT_places/imagesPlaces205_resize_val.lmdb')
    parser.add_argument('--debugtar', action='store_true', help='just show the images from tar')
    parser.add_argument('--debuglmdb', action='store_true', help='just show the images from lmdb')
    args = parser.parse_args()

    if args.debugtar:
        ds = PlaceReader(args.tar, args.labels)
        ds.reset_state()
        for jpeg, label in ds.get_data():
            rgb = cv2.imdecode(np.asarray(jpeg), cv2.IMREAD_COLOR)
            cv2.imshow("RGB image from Place2-dataset", rgb)
            print("label %i" % label)
            cv2.waitKey(0)
    elif args.debuglmdb:
        ds = LMDBDataPoint(args.lmdb, shuffle=True)
        ds.reset_state()
        for jpeg, label in ds.get_data():
            rgb = cv2.imdecode(np.asarray(jpeg), cv2.IMREAD_COLOR)
            cv2.imshow("RGB image from Place2-dataset", rgb)
            print("label %i" % label)
            cv2.waitKey(0)
    else:
        ds = PlaceReader(args.tar, args.labels)
        ds.reset_state()
        dftools.dump_dataflow_to_lmdb(ds, args.lmdb)
