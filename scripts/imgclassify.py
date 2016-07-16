#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: imgclassify.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>


import argparse
import cv2
import tensorflow as tf
import imp

from tensorpack.utils import *
from tensorpack.utils import sessinit
from tensorpack.dataflow import *
from tensorpack.predict import PredictConfig, SimpleDatasetPredictor


parser = argparse.ArgumentParser()
parser.add_argument(dest='config')
parser.add_argument(dest='model')
parser.add_argument(dest='images', nargs='+')
parser.add_argument('--output_type', default='label',
                    choices=['label', 'label-prob', 'raw'])
parser.add_argument('--top', default=1, type=int)
args = parser.parse_args()

get_config_func = imp.load_source('config_script', args.config).get_config

# TODO not sure if it this script is still working

with tf.Graph().as_default() as G:
    train_config = get_config_func()
    M = train_config.model
    config = PredictConfig(
        input_var_names=[M.get_input_vars_desc()[0].name],  # assume first component is image
        model=M,
        session_init=sessinit.SaverRestore(args.model),
        output_var_names=['output:0']
    )

    ds = ImageFromFile(args.images, 3, resize=(227, 227))
    ds = BatchData(ds, 128, remainder=True)
    predictor = SimpleDatasetPredictor(config, ds)
    res = predictor.get_all_result()

    if args.output_type == 'label':
        for r in res:
            print r[0].argsort(axis=1)[:,-args.top:][:,::-1]
    elif args.output_type == 'label_prob':
        raise NotImplementedError
    elif args.output_type == 'raw':
        print res
