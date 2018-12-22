#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: CAM-resnet.py

import argparse
import multiprocessing
import numpy as np
import os
import sys
import cv2
import tensorflow as tf

from tensorpack import *
from tensorpack.dataflow import dataset
from tensorpack.tfutils import gradproc, optimizer
from tensorpack.tfutils.summary import *
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.utils import viz
from tensorpack.utils.gpu import get_num_gpu

from imagenet_utils import ImageNetModel, fbresnet_augmentor
from resnet_model import preresnet_basicblock, preresnet_group

TOTAL_BATCH_SIZE = 256
DEPTH = None


class Model(ImageNetModel):

    def get_logits(self, image):
        cfg = {
            18: ([2, 2, 2, 2], preresnet_basicblock),
            34: ([3, 4, 6, 3], preresnet_basicblock),
        }
        defs, block_func = cfg[DEPTH]

        with argscope(Conv2D, use_bias=False,
                      kernel_initializer=tf.variance_scaling_initializer(scale=2.0, mode='fan_out')), \
                argscope([Conv2D, MaxPooling, GlobalAvgPooling, BatchNorm], data_format='channels_first'):
            convmaps = (LinearWrap(image)
                        .Conv2D('conv0', 64, 7, strides=2, activation=BNReLU)
                        .MaxPooling('pool0', 3, strides=2, padding='SAME')
                        .apply2(preresnet_group, 'group0', block_func, 64, defs[0], 1)
                        .apply2(preresnet_group, 'group1', block_func, 128, defs[1], 2)
                        .apply2(preresnet_group, 'group2', block_func, 256, defs[2], 2)
                        .apply2(preresnet_group, 'group3new', block_func, 512, defs[3], 1)())
            print(convmaps)
            convmaps = GlobalAvgPooling('gap', convmaps)
            logits = FullyConnected('linearnew', convmaps, 1000)
        return logits

    def optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=0.1, trainable=False)
        opt = tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True)
        gradprocs = [gradproc.ScaleGradient(
            [('conv0.*', 0.1), ('group[0-2].*', 0.1)])]
        return optimizer.apply_grad_processors(opt, gradprocs)


def get_data(train_or_test):
    # completely copied from imagenet-resnet.py example
    isTrain = train_or_test == 'train'

    datadir = args.data
    ds = dataset.ILSVRC12(datadir, train_or_test, shuffle=isTrain)
    augmentors = fbresnet_augmentor(isTrain)
    augmentors.append(imgaug.ToUint8())

    ds = AugmentImageComponent(ds, augmentors, copy=False)
    if isTrain:
        ds = PrefetchDataZMQ(ds, min(25, multiprocessing.cpu_count()))
    ds = BatchData(ds, BATCH_SIZE, remainder=not isTrain)
    return ds


def get_config():
    dataset_train = get_data('train')
    dataset_val = get_data('val')

    return TrainConfig(
        model=Model(),
        dataflow=dataset_train,
        callbacks=[
            ModelSaver(),
            PeriodicTrigger(InferenceRunner(dataset_val, [
                ClassificationError('wrong-top1', 'val-error-top1'),
                ClassificationError('wrong-top5', 'val-error-top5')]),
                every_k_epochs=2),
            ScheduledHyperParamSetter('learning_rate',
                                      [(30, 1e-2), (55, 1e-3), (75, 1e-4), (95, 1e-5)]),
        ],
        steps_per_epoch=5000,
        max_epoch=105,
    )


def viz_cam(model_file, data_dir):
    ds = get_data('val')
    pred_config = PredictConfig(
        model=Model(),
        session_init=get_model_loader(model_file),
        input_names=['input', 'label'],
        output_names=['wrong-top1', 'group3new/bnlast/Relu', 'linearnew/W'],
        return_input=True
    )
    meta = dataset.ILSVRCMeta().get_synset_words_1000()

    pred = SimpleDatasetPredictor(pred_config, ds)
    cnt = 0
    for inp, outp in pred.get_result():
        images, labels = inp
        wrongs, convmaps, W = outp
        batch = wrongs.shape[0]
        for i in range(batch):
            if wrongs[i]:
                continue
            weight = W[:, [labels[i]]].T    # 512x1
            convmap = convmaps[i, :, :, :]  # 512xhxw
            mergedmap = np.matmul(weight, convmap.reshape((512, -1))).reshape(14, 14)
            mergedmap = cv2.resize(mergedmap, (224, 224))
            heatmap = viz.intensity_to_rgb(mergedmap, normalize=True)
            blend = images[i] * 0.5 + heatmap * 0.5
            concat = np.concatenate((images[i], heatmap, blend), axis=1)

            classname = meta[labels[i]].split(',')[0]
            cv2.imwrite('cam{}-{}.jpg'.format(cnt, classname), concat)
            cnt += 1
            if cnt == 500:
                return


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--data', help='ILSVRC dataset dir')
    parser.add_argument('--depth', type=int, default=18)
    parser.add_argument('--load', help='load model')
    parser.add_argument('--cam', action='store_true', help='run visualization')
    args = parser.parse_args()

    DEPTH = args.depth
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    num_gpu = get_num_gpu()
    BATCH_SIZE = TOTAL_BATCH_SIZE // num_gpu

    if args.cam:
        BATCH_SIZE = 128    # something that can run on one gpu
        viz_cam(args.load, args.data)
        sys.exit()

    logger.auto_set_dir()
    config = get_config()
    if args.load:
        config.session_init = get_model_loader(args.load)
    launch_train_with_config(config, SyncMultiGPUTrainerParameterServer(num_gpu))
