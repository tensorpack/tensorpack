#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: checkpoint-prof.py

import argparse
import numpy as np
import tensorflow as tf

from tensorpack import get_default_sess_config, get_op_tensor_name
from tensorpack.tfutils.sessinit import SmartInit
from tensorpack.utils import logger

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='model file')
    parser.add_argument('--meta', help='metagraph proto file. Will be used to load the graph', required=True)
    parser.add_argument('-i', '--input', nargs='+', help='list of input tensors with their shapes.')
    parser.add_argument('-o', '--output', nargs='+', help='list of output tensors')
    parser.add_argument('--warmup', help='warmup iterations', type=int, default=5)
    parser.add_argument('--print-flops', action='store_true')
    parser.add_argument('--print-params', action='store_true')
    parser.add_argument('--print-timing', action='store_true')
    args = parser.parse_args()

    tf.train.import_meta_graph(args.meta, clear_devices=True)
    G = tf.get_default_graph()
    with tf.Session(config=get_default_sess_config()) as sess:
        init = SmartInit(args.model)
        init.init(sess)

        feed = {}
        for inp in args.input:
            inp = inp.split('=')
            name = get_op_tensor_name(inp[0].strip())[1]
            shape = list(map(int, inp[1].strip().split(',')))
            tensor = G.get_tensor_by_name(name)
            logger.info("Feeding shape ({}) to tensor {}".format(','.join(map(str, shape)), name))
            feed[tensor] = np.random.rand(*shape)

        fetches = []
        for name in args.output:
            name = get_op_tensor_name(name)[1]
            fetches.append(G.get_tensor_by_name(name))
        logger.info("Fetching tensors: {}".format(', '.join([k.name for k in fetches])))

        for _ in range(args.warmup):
            sess.run(fetches, feed_dict=feed)

        opt = tf.RunOptions()
        opt.trace_level = tf.RunOptions.FULL_TRACE
        meta = tf.RunMetadata()
        sess.run(fetches, feed_dict=feed, options=opt, run_metadata=meta)

        if args.print_flops:
            tf.profiler.profile(
                G,
                run_meta=meta,
                cmd='op',
                options=tf.profiler.ProfileOptionBuilder.float_operation())

        if args.print_params:
            tf.profiler.profile(
                G,
                run_meta=meta,
                options=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter())

        if args.print_timing:
            tf.profiler.profile(
                G,
                run_meta=meta,
                options=tf.profiler.ProfileOptionBuilder.time_and_memory())
