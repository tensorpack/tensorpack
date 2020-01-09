#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: dump-model-params.py

import argparse
import numpy as np
import os
import six
import tensorflow as tf

from tensorpack import logger
from tensorpack.tfutils import varmanip
from tensorpack.tfutils.common import get_op_tensor_name, get_tf_version_tuple

TF_version = get_tf_version_tuple()


def _import_external_ops(message):
    if "horovod" in message.lower():
        logger.info("Importing horovod ...")
        import horovod.tensorflow  # noqa
        return
    if "MaxBytesInUse" in message:
        logger.info("Importing memory_stats ...")
        from tensorflow.contrib.memory_stats import MaxBytesInUse  # noqa
        return
    if 'Nccl' in message:
        logger.info("Importing nccl ...")
        if TF_version <= (1, 12):
            try:
                from tensorflow.contrib.nccl.python.ops.nccl_ops import _validate_and_load_nccl_so
            except Exception:
                pass
            else:
                _validate_and_load_nccl_so()
            from tensorflow.contrib.nccl.ops import gen_nccl_ops  # noqa
        else:
            from tensorflow.python.ops import gen_nccl_ops  # noqa
        return


def guess_inputs(input_dir):
    meta_candidates = []
    model_candidates = []
    for path in os.listdir(input_dir):
        if path.startswith('graph-') and path.endswith('.meta'):
            meta_candidates.append(path)
        if path.startswith('model-') and path.endswith('.index'):
            modelid = int(path[len('model-'):-len('.index')])
            model_candidates.append((path, modelid))
    assert len(meta_candidates)
    meta = sorted(meta_candidates)[-1]
    if len(meta_candidates) > 1:
        logger.info("Choosing {} from {} as graph file.".format(meta, meta_candidates))
    else:
        logger.info("Choosing {} as graph file.".format(meta))

    assert len(model_candidates)
    model = sorted(model_candidates, key=lambda x: x[1])[-1][0]
    if len(model_candidates) > 1:
        logger.info("Choosing {} from {} as model file.".format(model, [x[0] for x in model_candidates]))
    else:
        logger.info("Choosing {} as model file.".format(model))
    return os.path.join(input_dir, model), os.path.join(input_dir, meta)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Keep only TRAINABLE and MODEL variables in a checkpoint.')
    parser.add_argument('--meta', help='metagraph file')
    parser.add_argument(dest='input', help='input model file, has to be a TF checkpoint')
    parser.add_argument(dest='output', help='output model file, can be npz or TF checkpoint')
    args = parser.parse_args()

    if os.path.isdir(args.input):
        input, meta = guess_inputs(args.input)
    else:
        meta = args.meta
        input = args.input

    # this script does not need GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = ''

    if args.meta is not None:
        while True:
            try:
                tf.reset_default_graph()
                tf.train.import_meta_graph(meta, clear_devices=True)
            except KeyError as e:
                op_name = e.args[0]
                _import_external_ops(op_name)
            except tf.errors.NotFoundError as e:
                _import_external_ops(str(e))
            else:
                break

    # loading...
    if input.endswith('.npz'):
        dic = np.load(input)
    else:
        dic = varmanip.load_chkpt_vars(input)
    dic = {get_op_tensor_name(k)[1]: v for k, v in six.iteritems(dic)}

    if args.meta is not None:
        # save variables that are GLOBAL, and either TRAINABLE or MODEL
        var_to_dump = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        var_to_dump.extend(tf.get_collection(tf.GraphKeys.MODEL_VARIABLES))
        if len(set(var_to_dump)) != len(var_to_dump):
            logger.warn("TRAINABLE and MODEL variables have duplication!")
        var_to_dump = list(set(var_to_dump))
        globvarname = {k.name for k in tf.global_variables()}
        var_to_dump = {k.name for k in var_to_dump if k.name in globvarname}

        for name in var_to_dump:
            assert name in dic, "Variable {} not found in the model!".format(name)
    else:
        var_to_dump = set(dic.keys())

    dic_to_dump = {k: v for k, v in six.iteritems(dic) if k in var_to_dump}
    varmanip.save_chkpt_vars(dic_to_dump, args.output)
