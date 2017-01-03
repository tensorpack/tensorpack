#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: dump-dataflow.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import argparse
import cv2
import imp
import tqdm
import os
from tensorpack.utils import logger
from tensorpack.utils.fs import mkdir_p
from tensorpack.dataflow import RepeatedData


parser = argparse.ArgumentParser()
parser.add_argument(dest='config')
parser.add_argument('-o', '--output',
                    help='output directory to dump dataset image. If not given, will not dump images.')
parser.add_argument('-s', '--scale',
                    help='scale the image data (maybe by 255)', default=1, type=int)
parser.add_argument('--index',
                    help='index of the image component in datapoint',
                    default=0, type=int)
parser.add_argument('-n', '--number', help='number of images to dump',
                    default=10, type=int)
args = parser.parse_args()
logger.auto_set_dir(action='d')

get_config_func = imp.load_source('config_script', args.config).get_config
config = get_config_func()
config.dataset.reset_state()

if args.output:
    mkdir_p(args.output)
    cnt = 0
    index = args.index   # TODO: as an argument?
    for dp in config.dataset.get_data():
        imgbatch = dp[index]
        if cnt > args.number:
            break
        for bi, img in enumerate(imgbatch):
            cnt += 1
            fname = os.path.join(args.output, '{:03d}-{}.png'.format(cnt, bi))
            cv2.imwrite(fname, img * args.scale)

NR_DP_TEST = args.number
logger.info("Testing dataflow speed:")
ds = RepeatedData(config.dataset, -1)
with tqdm.tqdm(total=NR_DP_TEST, leave=True, unit='data points') as pbar:
    for idx, dp in enumerate(ds.get_data()):
        del dp
        if idx > NR_DP_TEST:
            break
        pbar.update()
