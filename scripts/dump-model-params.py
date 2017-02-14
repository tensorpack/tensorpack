#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: dump-model-params.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import numpy as np
import argparse
import tensorflow as tf
import imp

from tensorpack import TowerContext, logger, ModelFromMetaGraph
from tensorpack.tfutils import sessinit, varmanip

parser = argparse.ArgumentParser()
parser.add_argument('--config', help='config file')
parser.add_argument('--meta', help='metagraph file')
parser.add_argument(dest='model')
parser.add_argument(dest='output')
args = parser.parse_args()

assert args.config or args.meta, "Either config or metagraph must be present!"

with tf.Graph().as_default() as G:
    if args.config:
        MODEL = imp.load_source('config_script', args.config).Model
        M = MODEL()
        with TowerContext('', is_training=False):
            M.build_graph(M.get_reused_placehdrs())
    else:
        M = ModelFromMetaGraph(args.meta)

    # loading...
    if args.model.endswith('.npy'):
        init = sessinit.ParamRestore(np.load(args.model).item())
    else:
        init = sessinit.SaverRestore(args.model)
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    sess.run(tf.global_variables_initializer())
    init.init(sess)

    # dump ...
    with sess.as_default():
        if args.output.endswith('npy'):
            varmanip.dump_session_params(args.output)
        else:
            var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            var.extend(tf.get_collection(tf.GraphKeys.MODEL_VARIABLES))
            var_dict = {}
            for v in var:
                name = varmanip.get_savename_from_varname(v.name)
                var_dict[name] = v
            logger.info("Variables to dump:")
            logger.info(", ".join(var_dict.keys()))
            saver = tf.train.Saver(
                var_list=var_dict,
                write_version=tf.train.SaverDef.V2)
            saver.save(sess, args.output, write_meta_graph=False)
