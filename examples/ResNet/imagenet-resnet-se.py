#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: imagenet-resnet-se.py

import sys
import argparse
import numpy as np
import os

import tensorflow as tf

from tensorpack import *
from tensorpack.dataflow import dataset
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.tfutils.summary import *
from tensorpack.utils.gpu import get_nr_gpu

from imagenet_resnet_utils import (
    fbresnet_augmentor, apply_preactivation, resnet_shortcut, resnet_backbone,
    resnet_group, eval_on_ILSVRC12, image_preprocess, compute_loss_and_error,
    get_bn,
    get_imagenet_dataflow)

TOTAL_BATCH_SIZE = 256
INPUT_SHAPE = 224
DEPTH = None

RESNET_CONFIG = {
    50: [3, 4, 6, 3],
    101: [3, 4, 23, 3],
}


class Model(ModelDesc):
    def _get_inputs(self):
        return [InputDesc(tf.float32, [None, INPUT_SHAPE, INPUT_SHAPE, 3], 'input'),
                InputDesc(tf.int32, [None], 'label')]

    def _build_graph(self, inputs):
        image, label = inputs
        image = image_preprocess(image, bgr=True)
        image = tf.transpose(image, [0, 3, 1, 2])

        def bottleneck_se(l, ch_out, stride, preact):
            l, shortcut = apply_preactivation(l, preact)
            l = Conv2D('conv1', l, ch_out, 1, nl=BNReLU)
            l = Conv2D('conv2', l, ch_out, 3, stride=stride, nl=BNReLU)
            l = Conv2D('conv3', l, ch_out * 4, 1, nl=get_bn(zero_init=True))

            squeeze = GlobalAvgPooling('gap', l)
            squeeze = FullyConnected('fc1', squeeze, ch_out // 4, nl=tf.nn.relu)
            squeeze = FullyConnected('fc2', squeeze, ch_out * 4, nl=tf.nn.sigmoid)
            l = l * tf.reshape(squeeze, [-1, ch_out * 4, 1, 1])
            return l + resnet_shortcut(shortcut, ch_out * 4, stride, nl=get_bn(zero_init=False))
        defs = RESNET_CONFIG[DEPTH]

        with argscope([Conv2D, MaxPooling, GlobalAvgPooling, BatchNorm], data_format='NCHW'):
            logits = resnet_backbone(image, defs, resnet_group, bottleneck_se)

        loss = compute_loss_and_error(logits, label)
        wd_loss = regularize_cost('.*/W', l2_regularizer(1e-4), name='l2_regularize_loss')
        add_moving_summary(loss, wd_loss)
        self.cost = tf.add_n([loss, wd_loss], name='cost')

    def _get_optimizer(self):
        lr = get_scalar_var('learning_rate', 0.1, summary=True)
        return tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True)


def get_data(name, batch):
    isTrain = name == 'train'
    augmentors = fbresnet_augmentor(isTrain)
    datadir = args.data
    return get_imagenet_dataflow(
        datadir, name, batch, augmentors)


def get_config():
    assert tf.test.is_gpu_available()
    nr_gpu = get_nr_gpu()
    batch = TOTAL_BATCH_SIZE // nr_gpu
    logger.info("Running on {} GPUs. Batch size per GPU: {}".format(nr_gpu, batch))

    dataset_train = get_data('train', batch)
    dataset_val = get_data('val', batch)

    callbacks = [
        ModelSaver(),
        ScheduledHyperParamSetter('learning_rate',
                                  [(30, 1e-2), (60, 1e-3), (85, 1e-4), (95, 1e-5), (105, 1e-6)]),
        HumanHyperParamSetter('learning_rate'),
    ]
    infs = [ClassificationError('wrong-top1', 'val-error-top1'),
            ClassificationError('wrong-top5', 'val-error-top5')]
    if nr_tower == 1:
        callbacks.append(InferenceRunner(QueueInput(dataset_val), infs))
    else:
        callbacks.append(DataParallelInferenceRunner(
            dataset_val, infs, list(range(nr_tower))))

    return TrainConfig(
        model=Model(),
        dataflow=dataset_train,
        callbacks=callbacks,
        steps_per_epoch=5000,
        max_epoch=110,
        nr_tower=nr_tower
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--data', help='ILSVRC dataset dir')
    parser.add_argument('--load', help='load model')
    parser.add_argument('-d', '--depth', help='resnet depth',
                        type=int, default=50, choices=[50, 101])
    parser.add_argument('--eval', action='store_true')
    args = parser.parse_args()

    DEPTH = args.depth
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if args.eval:
        ds = get_data('val', 128)
        eval_on_ILSVRC12(Model(), get_model_loader(args.load), ds)
        sys.exit()

    logger.set_logger_dir(
        os.path.join('train_log', 'imagenet-resnet-se-d' + str(DEPTH)))

    config = get_config(Model())
    if args.load:
        config.session_init = SaverRestore(args.load)
    SyncMultiGPUTrainerParameterServer(config).train()
