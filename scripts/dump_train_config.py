#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: dump_train_config.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import argparse
import cv2
import tensorflow as tf
import imp
import os
from tensorpack.utils.utils import mkdir_p

parser = argparse.ArgumentParser()
parser.add_argument(dest='config')
parser.add_argument(dest='output')
parser.add_argument('-n', '--number', help='number of images to take',
                    default=10, type=int)
args = parser.parse_args()

mkdir_p(args.output)

index = 0   # TODO: as an argument?

get_config_func = imp.load_source('config_script', args.config).get_config
config = get_config_func()

cnt = 0
for dp in config.dataset.get_data():
    imgbatch = dp[index]
    if cnt > args.number:
        break
    for bi, img in enumerate(imgbatch):
        cnt += 1
        fname = os.path.join(args.output, '{:03d}-{}.png'.format(cnt, bi))
        cv2.imwrite(fname, img * 255.0)
