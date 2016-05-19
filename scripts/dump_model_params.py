#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# File: dump_model_params.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import argparse
import cv2
import tensorflow as tf
import imp

from tensorpack.utils import *
from tensorpack.tfutils import sessinit
from tensorpack.dataflow import *

parser = argparse.ArgumentParser()
parser.add_argument(dest='config')
parser.add_argument(dest='model')
parser.add_argument(dest='output')
args = parser.parse_args()

get_config_func = imp.load_source('config_script', args.config).get_config

with tf.Graph().as_default() as G:
    config = get_config_func()
    config.model.build_graph(config.model.get_input_vars(), is_training=False)
    init = sessinit.SaverRestore(args.model)
    sess = tf.Session()
    init.init(sess)
    with sess.as_default():
        sessinit.dump_session_params(args.output)
