#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: dump-model-params.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>

import argparse
import tensorflow as tf

from tensorpack import logger
from tensorpack.tfutils import varmanip, get_model_loader

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Keep only TRAINABLE and MODEL variables in a checkpoint.')
    parser.add_argument('--meta', help='metagraph file', required=True)
    parser.add_argument(dest='input', help='input model file, has to be a TF checkpoint')
    parser.add_argument(dest='output', help='output model file, can be npz or TF checkpoint')
    args = parser.parse_args()

    tf.train.import_meta_graph(args.meta, clear_devices=True)

    # loading...
    init = get_model_loader(args.input)
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    init.init(sess)

    # dump ...
    with sess.as_default():
        if args.output.endswith('npy') or args.output.endswith('npz'):
            varmanip.dump_session_params(args.output)
        else:
            var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
            var.extend(tf.get_collection(tf.GraphKeys.MODEL_VARIABLES))
            gvars = set([k.name for k in tf.global_variables()])
            var = [v for v in var if v.name in gvars]
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
