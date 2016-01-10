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
from tensorpack.predict import PredictConfig, DatasetPredictor


parser = argparse.ArgumentParser()
parser.add_argument(dest='config')
parser.add_argument(dest='model')
parser.add_argument(dest='images', nargs='+')
parser.add_argument('--output_type', default='label',
                    choices=['label', 'label-prob', 'raw'])
parser.add_argument('--top', default=1, type=int)
args = parser.parse_args()

get_config_func = imp.load_source('config_script', args.config).get_config

with tf.Graph().as_default() as G:
    train_config = get_config_func()
    config = PredictConfig(
        inputs=train_config.inputs,
        input_dataset_mapping=[train_config.inputs[0]],  # assume first component is image
        get_model_func=train_config.get_model_func,
        session_init=sessinit.SaverRestore(args.model),
        output_var_names=['output:0']
    )

    ds = ImageFromFile(args.images, 3, resize=(227, 227))
    predictor = DatasetPredictor(config, ds, batch=128)
    res = predictor.get_all_result()
    res = [k.output for k in res]

    if args.output_type == 'label':
        for r in res:
            print r[0].argsort(axis=1)[:,-args.top:][:,::-1]
    elif args.output_type == 'label_prob':
        raise NotImplementedError
    elif args.output_type == 'raw':
        print res
