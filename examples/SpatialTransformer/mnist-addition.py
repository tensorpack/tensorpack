#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: mnist-addition.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import numpy as np
import tensorflow as tf
import os
import sys
import cv2
import argparse

from tensorpack import *
import tensorpack.tfutils.symbolic_functions as symbf

IMAGE_SIZE = 42
WARP_TARGET_SIZE = 28
HALF_DIFF = (IMAGE_SIZE - WARP_TARGET_SIZE) // 2


class Model(ModelDesc):
    def _get_inputs(self):
        return [InputDesc(tf.float32, (None, IMAGE_SIZE, IMAGE_SIZE, 2), 'input'),
                InputDesc(tf.int32, (None,), 'label')]

    def _build_graph(self, inputs):
        xys = np.array([(y, x, 1) for y in range(WARP_TARGET_SIZE)
                        for x in range(WARP_TARGET_SIZE)], dtype='float32')
        xys = tf.constant(xys, dtype=tf.float32, name='xys')    # p x 3

        image, label = inputs

        image = image / 255.0 - 0.5  # bhw2

        def get_stn(image):
            stn = (LinearWrap(image)
                   .AvgPooling('downsample', 2)
                   .Conv2D('conv0', 20, 5, padding='VALID')
                   .MaxPooling('pool0', 2)
                   .Conv2D('conv1', 20, 5, padding='VALID')
                   .FullyConnected('fc1', out_dim=32)
                   .FullyConnected('fct', out_dim=6, nl=tf.identity,
                                   W_init=tf.constant_initializer(),
                                   b_init=tf.constant_initializer([1, 0, HALF_DIFF, 0, 1, HALF_DIFF]))())
            # output 6 parameters for affine transformation
            stn = tf.reshape(stn, [-1, 2, 3], name='affine')  # bx2x3
            stn = tf.reshape(tf.transpose(stn, [2, 0, 1]), [3, -1])  # 3 x (bx2)
            coor = tf.reshape(tf.matmul(xys, stn),
                              [WARP_TARGET_SIZE, WARP_TARGET_SIZE, -1, 2])
            coor = tf.transpose(coor, [2, 0, 1, 3], 'sampled_coords')  # b h w 2
            sampled = ImageSample('warp', [image, coor], borderMode='constant')
            return sampled

        with argscope([Conv2D, FullyConnected], nl=tf.nn.relu):
            with tf.variable_scope('STN1'):
                sampled1 = get_stn(image)
            with tf.variable_scope('STN2'):
                sampled2 = get_stn(image)

        # For visualization in tensorboard
        padded1 = tf.pad(sampled1, [[0, 0], [HALF_DIFF, HALF_DIFF], [HALF_DIFF, HALF_DIFF], [0, 0]])
        padded2 = tf.pad(sampled2, [[0, 0], [HALF_DIFF, HALF_DIFF], [HALF_DIFF, HALF_DIFF], [0, 0]])
        img_orig = tf.concat([image[:, :, :, 0], image[:, :, :, 1]], 1)  # b x 2h  x w
        transform1 = tf.concat([padded1[:, :, :, 0], padded1[:, :, :, 1]], 1)
        transform2 = tf.concat([padded2[:, :, :, 0], padded2[:, :, :, 1]], 1)
        stacked = tf.concat([img_orig, transform1, transform2], 2, 'viz')
        tf.summary.image('visualize',
                         tf.expand_dims(stacked, -1), max_outputs=30)

        sampled = tf.concat([sampled1, sampled2], 3, 'sampled_concat')
        logits = (LinearWrap(sampled)
                  .FullyConnected('fc1', out_dim=256, nl=tf.nn.relu)
                  .FullyConnected('fc2', out_dim=128, nl=tf.nn.relu)
                  .FullyConnected('fct', out_dim=19, nl=tf.identity)())
        prob = tf.nn.softmax(logits, name='prob')

        cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label)
        cost = tf.reduce_mean(cost, name='cross_entropy_loss')

        wrong = symbolic_functions.prediction_incorrect(logits, label)
        summary.add_moving_summary(tf.reduce_mean(wrong, name='train_error'))

        wd_cost = tf.multiply(1e-5, regularize_cost('fc.*/W', tf.nn.l2_loss),
                              name='regularize_loss')
        summary.add_moving_summary(cost, wd_cost)
        self.cost = tf.add_n([wd_cost, cost], name='cost')

    def _get_optimizer(self):
        lr = symbf.get_scalar_var('learning_rate', 5e-4, summary=True)
        opt = tf.train.AdamOptimizer(lr, epsilon=1e-3)
        return optimizer.apply_grad_processors(
            opt, [
                gradproc.ScaleGradient(('STN.*', 0.1)),
                gradproc.SummaryGradient()])


def get_data(isTrain):
    ds = dataset.Mnist('train' if isTrain else 'test')
    # create augmentation for both training and testing
    augs = [
        imgaug.MapImage(lambda x: x * 255.0),
        imgaug.RandomResize((0.7, 1.2), (0.7, 1.2)),
        imgaug.RotationAndCropValid(45),
        imgaug.RandomPaste((IMAGE_SIZE, IMAGE_SIZE)),
        imgaug.SaltPepperNoise(white_prob=0.01, black_prob=0.01)
    ]
    ds = AugmentImageComponent(ds, augs)

    ds = JoinData([ds, ds])
    # stack the two digits into two channels, and label it with the sum
    ds = MapData(ds, lambda dp: [np.stack([dp[0], dp[2]], axis=2), dp[1] + dp[3]])
    ds = BatchData(ds, 128)
    return ds


def view_warp(modelpath):
    pred = OfflinePredictor(PredictConfig(
        session_init=get_model_loader(modelpath),
        model=Model(),
        input_names=['input'],
        output_names=['viz', 'STN1/affine', 'STN2/affine']))

    xys = np.array([[0, 0, 1],
                    [WARP_TARGET_SIZE, 0, 1],
                    [WARP_TARGET_SIZE, WARP_TARGET_SIZE, 1],
                    [0, WARP_TARGET_SIZE, 1]], dtype='float32')

    def draw_rect(img, affine, c, offset=[0, 0]):
        a = np.transpose(affine)  # 3x2
        a = (np.matmul(xys, a) + offset).astype('int32')
        cv2.line(img, tuple(a[0][::-1]), tuple(a[1][::-1]), c)
        cv2.line(img, tuple(a[1][::-1]), tuple(a[2][::-1]), c)
        cv2.line(img, tuple(a[2][::-1]), tuple(a[3][::-1]), c)
        cv2.line(img, tuple(a[3][::-1]), tuple(a[0][::-1]), c)

    ds = get_data(False)
    ds.reset_state()
    for k in ds.get_data():
        img, label = k
        outputs, affine1, affine2 = pred([img])
        for idx, viz in enumerate(outputs):
            viz = cv2.cvtColor(viz, cv2.COLOR_GRAY2BGR)
            # Here we assume the second branch focuses on the first digit
            draw_rect(viz, affine2[idx], (0, 0, 255))
            draw_rect(viz, affine1[idx], (0, 0, 255), offset=[IMAGE_SIZE, 0])
            cv2.imwrite('{:03d}.png'.format(idx), (viz + 0.5) * 255)
        break


def get_config():
    logger.auto_set_dir()

    dataset_train, dataset_test = get_data(True), get_data(False)
    steps_per_epoch = dataset_train.size() * 5

    return TrainConfig(
        model=Model(),
        dataflow=dataset_train,
        callbacks=[
            ModelSaver(),
            InferenceRunner(dataset_test,
                            [ScalarStats('cost'), ClassificationError()]),
            ScheduledHyperParamSetter('learning_rate', [(200, 1e-4)])
        ],
        session_config=get_default_sess_config(0.5),
        steps_per_epoch=steps_per_epoch,
        max_epoch=500,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--load', help='load model')
    parser.add_argument('--view', action='store_true')
    args = parser.parse_args()
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    if args.view:
        view_warp(args.load)
    else:
        config = get_config()
        if args.load:
            config.session_init = SaverRestore(args.load)
        SimpleTrainer(config).train()
