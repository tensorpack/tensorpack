#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: imagenet-resnet-keras.py
# Author: Yuxin Wu

import argparse
import numpy as np
import os
import tensorflow as tf
from tensorflow.python.keras.layers import *

from tensorpack import InputDesc, SyncMultiGPUTrainerReplicated
from tensorpack.callbacks import *
from tensorpack.contrib.keras import KerasModel
from tensorpack.dataflow import FakeData, MapDataComponent
from tensorpack.tfutils.common import get_tf_version_tuple
from tensorpack.utils import logger
from tensorpack.utils.gpu import get_num_gpu

from imagenet_utils import fbresnet_augmentor, get_imagenet_dataflow

TOTAL_BATCH_SIZE = 512
BASE_LR = 0.1 * (TOTAL_BATCH_SIZE // 256)


def bn(x, name, zero_init=False):
    return BatchNormalization(
        axis=1, name=name, fused=True,
        momentum=0.9, epsilon=1e-5,
        gamma_initializer='zeros' if zero_init else 'ones')(x)


def conv(x, filters, kernel, strides=1, name=None):
    return Conv2D(filters, kernel, name=name,
                  strides=strides, use_bias=False, padding='same',
                  kernel_initializer=tf.keras.initializers.VarianceScaling(
                      scale=2.0, mode='fan_out',
                      distribution='untruncated_normal' if get_tf_version_tuple() >= (1, 12) else 'normal'),
                  kernel_regularizer=tf.keras.regularizers.l2(5e-5))(x)


def identity_block(input_tensor, kernel_size, filters, stage, block):
    filters1, filters2, filters3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = conv(input_tensor, filters1, 1, name=conv_name_base + '2a')
    x = bn(x, name=bn_name_base + '2a')
    x = Activation('relu')(x)

    x = conv(x, filters2, kernel_size, name=conv_name_base + '2b')
    x = bn(x, name=bn_name_base + '2b')
    x = Activation('relu')(x)

    x = conv(x, filters3, (1, 1), name=conv_name_base + '2c')
    x = bn(x, name=bn_name_base + '2c', zero_init=True)

    x = tf.keras.layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    filters1, filters2, filters3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = conv(input_tensor, filters1, (1, 1), name=conv_name_base + '2a')
    x = bn(x, name=bn_name_base + '2a')
    x = Activation('relu')(x)

    x = conv(x, filters2, kernel_size, strides=strides, name=conv_name_base + '2b')
    x = bn(x, name=bn_name_base + '2b')
    x = Activation('relu')(x)

    x = conv(x, filters3, (1, 1), name=conv_name_base + '2c')
    x = bn(x, name=bn_name_base + '2c', zero_init=True)

    shortcut = conv(
        input_tensor,
        filters3, (1, 1), strides=strides,
        name=conv_name_base + '1')
    shortcut = bn(shortcut, name=bn_name_base + '1')

    x = tf.keras.layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x


def resnet50(image):
    input = Input(tensor=image)

    def image_preprocess(image):
        image = tf.cast(image, tf.float32)
        image = image * (1.0 / 255)
        mean = [0.485, 0.456, 0.406][::-1]
        std = [0.229, 0.224, 0.225][::-1]
        image = (image - tf.constant(mean, dtype=tf.float32)) / tf.constant(std, dtype=tf.float32)
        image = tf.transpose(image, [0, 3, 1, 2])
        return image

    x = Lambda(image_preprocess)(input)

    x = conv(x, 64, (7, 7), strides=(2, 2), name='conv0')
    x = bn(x, name='bn_conv1')
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    x = GlobalAveragePooling2D(name='avg_pool')(x)
    x = Flatten()(x)
    x = Dense(1000, activation='softmax', name='fc1000',
              kernel_initializer=tf.keras.initializers.VarianceScaling(
                  scale=2.0, mode='fan_in'),
              kernel_regularizer=tf.keras.regularizers.l2(5e-5))(x)

    M = tf.keras.models.Model(input, x, name='resnet50')
    return M


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', help='ILSVRC dataset dir')
    parser.add_argument('--fake', help='use fakedata to test or benchmark this model', action='store_true')
    args = parser.parse_args()
    logger.set_logger_dir(os.path.join("train_log", "imagenet-resnet-keras"))

    tf.keras.backend.set_image_data_format('channels_first')

    num_gpu = get_num_gpu()
    if args.fake:
        df_train = FakeData([[32, 224, 224, 3], [32, 1000]], 5000, random=False, dtype='uint8')
        df_val = FakeData([[32, 224, 224, 3], [32, 1000]], 5000, random=False)
    else:
        batch_size = TOTAL_BATCH_SIZE // num_gpu
        assert args.data is not None
        df_train = get_imagenet_dataflow(
            args.data, 'train', batch_size, fbresnet_augmentor(True))
        df_val = get_imagenet_dataflow(
            args.data, 'val', batch_size, fbresnet_augmentor(False))

        def one_hot(label):
            return np.eye(1000)[label]

        df_train = MapDataComponent(df_train, one_hot, 1)
        df_val = MapDataComponent(df_val, one_hot, 1)

    M = KerasModel(
        resnet50,
        inputs_desc=[InputDesc(tf.uint8, [None, 224, 224, 3], 'images')],
        targets_desc=[InputDesc(tf.float32, [None, 1000], 'labels')],
        input=df_train,
        trainer=SyncMultiGPUTrainerReplicated(num_gpu))

    lr = tf.get_variable('learning_rate', initializer=0.1, trainable=False)
    tf.summary.scalar('lr', lr)

    M.compile(
        optimizer=tf.train.MomentumOptimizer(lr, 0.9, use_nesterov=True),
        loss='categorical_crossentropy',
        metrics='categorical_accuracy'
    )

    callbacks = [
        ModelSaver(),
        ScheduledHyperParamSetter(
            'learning_rate',
            [(0, 0.1), (3, BASE_LR)], interp='linear'),  # warmup
        ScheduledHyperParamSetter(
            'learning_rate',
            [(30, BASE_LR * 0.1), (60, BASE_LR * 1e-2), (85, BASE_LR * 1e-3)]),
        GPUUtilizationTracker()
    ]
    if not args.fake:
        callbacks.append(
            DataParallelInferenceRunner(
                df_val, ScalarStats(['categorical_accuracy']), num_gpu))

    M.fit(
        steps_per_epoch=100 if args.fake else 1281167 // TOTAL_BATCH_SIZE,
        max_epoch=100,
        callbacks=callbacks
    )
