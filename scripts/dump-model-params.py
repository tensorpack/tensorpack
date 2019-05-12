#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: dump-model-params.py

import argparse
import sys
import numpy as np
import os
import six
import tensorflow as tf

from tensorpack.tfutils import varmanip
from tensorpack.tfutils.common import get_op_tensor_name


def _import_external_ops(op_name):
    if "horovod" in op_name.lower():
        import horovod.tensorflow  # noqa
        return
    if op_name == "MaxBytesInUse":
        from tensorflow.contrib.memory_stats import MaxBytesInUse  # noqa
        return
    print("Your graph contains op '{}' which is not loaded into your Tensorflow runtime.".format(op_name))
    print("Therefore the graph cannot be loaded unless you import the relevant libraries first.")
    sys.exit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Keep only TRAINABLE and MODEL variables in a checkpoint.')
    parser.add_argument('--meta', help='metagraph file', required=True)
    parser.add_argument(dest='input', help='input model file, has to be a TF checkpoint')
    parser.add_argument(dest='output', help='output model file, can be npz or TF checkpoint')
    args = parser.parse_args()

    # this script does not need GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = ''

    while True:
        try:
            tf.reset_default_graph()
            tf.train.import_meta_graph(args.meta, clear_devices=True)
        except KeyError as e:
            op_name = e.args[0]
            _import_external_ops(op_name)
        else:
            break

    # loading...
    if args.input.endswith('.npz'):
        dic = np.load(args.input)
    else:
        dic = varmanip.load_chkpt_vars(args.input)
    dic = {get_op_tensor_name(k)[1]: v for k, v in six.iteritems(dic)}

    # save variables that are GLOBAL, and either TRAINABLE or MODEL
    var_to_dump = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    var_to_dump.extend(tf.get_collection(tf.GraphKeys.MODEL_VARIABLES))
    if len(set(var_to_dump)) != len(var_to_dump):
        print("TRAINABLE and MODEL variables have duplication!")
    var_to_dump = list(set(var_to_dump))
    globvarname = set([k.name for k in tf.global_variables()])
    var_to_dump = set([k.name for k in var_to_dump if k.name in globvarname])

    for name in var_to_dump:
        assert name in dic, "Variable {} not found in the model!".format(name)

    dic_to_dump = {k: v for k, v in six.iteritems(dic) if k in var_to_dump}
    varmanip.save_chkpt_vars(dic_to_dump, args.output)
