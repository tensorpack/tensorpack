#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: cifar-convnet.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>
import tensorflow as tf
import argparse
import os


from tensorpack import *
from tensorpack.tfutils.summary import *
from tensorpack.dataflow import dataset

"""
A small convnet model for Cifar10 or Cifar100 dataset.

Cifar10 trained on 1 GPU:
    91% accuracy after 50k iterations.
    70 itr/s on P100

Not a good model for Cifar100, just for demonstration.
"""


class Model(ModelDesc):
    def __init__(self, cifar_classnum):
        super(Model, self).__init__()
        self.cifar_classnum = cifar_classnum

    def _get_inputs(self):
        return [InputDesc(tf.float32, (None, 30, 30, 3), 'input'),
                InputDesc(tf.int32, (None,), 'label')
                ]

    def _build_graph(self, inputs):
        image, label = inputs
        is_training = get_current_tower_context().is_training
        keep_prob = tf.constant(0.5 if is_training else 1.0)

        if is_training:
            tf.summary.image("train_image", image, 10)
        if tf.test.is_gpu_available():
            image = tf.transpose(image, [0, 3, 1, 2])
            data_format = 'NCHW'
        else:
            data_format = 'NHWC'

        image = image / 4.0     # just to make range smaller
        with argscope(Conv2D, nl=BNReLU, use_bias=False, kernel_shape=3), \
                argscope([Conv2D, MaxPooling, BatchNorm], data_format=data_format):
            logits = LinearWrap(image) \
                .Conv2D('conv1.1', out_channel=64) \
                .Conv2D('conv1.2', out_channel=64) \
                .MaxPooling('pool1', 3, stride=2, padding='SAME') \
                .Conv2D('conv2.1', out_channel=128) \
                .Conv2D('conv2.2', out_channel=128) \
                .MaxPooling('pool2', 3, stride=2, padding='SAME') \
                .Conv2D('conv3.1', out_channel=128, padding='VALID') \
                .Conv2D('conv3.2', out_channel=128, padding='VALID') \
                .FullyConnected('fc0', 1024 + 512, nl=tf.nn.relu) \
                .tf.nn.dropout(keep_prob) \
                .FullyConnected('fc1', 512, nl=tf.nn.relu) \
                .FullyConnected('linear', out_dim=self.cifar_classnum, nl=tf.identity)()

        cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label)
        cost = tf.reduce_mean(cost, name='cross_entropy_loss')

        correct = tf.to_float(tf.nn.in_top_k(logits, label, 1), name='correct')
        # monitor training error
        add_moving_summary(tf.reduce_mean(correct, name='accuracy'))

        # weight decay on all W of fc layers
        wd_cost = regularize_cost('fc.*/W', l2_regularizer(4e-4), name='regularize_loss')
        add_moving_summary(cost, wd_cost)

        add_param_summary(('.*/W', ['histogram']))   # monitor W
        self.cost = tf.add_n([cost, wd_cost], name='cost')

    def _get_optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=1e-2, trainable=False)
        tf.summary.scalar('lr', lr)
        return tf.train.AdamOptimizer(lr, epsilon=1e-3)


def get_data(train_or_test, cifar_classnum):
    isTrain = train_or_test == 'train'
    if cifar_classnum == 10:
        ds = dataset.Cifar10(train_or_test)
    else:
        ds = dataset.Cifar100(train_or_test)
    if isTrain:
        augmentors = [
            imgaug.RandomCrop((30, 30)),
            imgaug.Flip(horiz=True),
            imgaug.Brightness(63),
            imgaug.Contrast((0.2, 1.8)),
            imgaug.MeanVarianceNormalize(all_channel=True)
        ]
    else:
        augmentors = [
            imgaug.CenterCrop((30, 30)),
            imgaug.MeanVarianceNormalize(all_channel=True)
        ]
    ds = AugmentImageComponent(ds, augmentors)
    ds = BatchData(ds, 128, remainder=not isTrain)
    if isTrain:
        ds = PrefetchDataZMQ(ds, 5)
    return ds


def get_config(cifar_classnum):
    # prepare dataset
    dataset_train = get_data('train', cifar_classnum)
    dataset_test = get_data('test', cifar_classnum)

    def lr_func(lr):
        if lr < 3e-5:
            raise StopTraining()
        return lr * 0.31
    return TrainConfig(
        model=Model(cifar_classnum),
        dataflow=dataset_train,
        callbacks=[
            ModelSaver(),
            InferenceRunner(dataset_test,
                            ScalarStats(['accuracy', 'cost'])),
            StatMonitorParamSetter('learning_rate', 'val_error', lr_func,
                                   threshold=0.001, last_k=10),
        ],
        max_epoch=150,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.', required=True)
    parser.add_argument('--load', help='load model')
    parser.add_argument('--classnum', help='10 for cifar10 or 100 for cifar100',
                        type=int, default=10)
    args = parser.parse_args()

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    with tf.Graph().as_default():
        logger.set_logger_dir(os.path.join('train_log', 'cifar' + str(args.classnum)))
        config = get_config(args.classnum)
        if args.load:
            config.session_init = SaverRestore(args.load)

        nr_gpu = len(args.gpu.split(','))
        trainer = QueueInputTrainer() if nr_gpu <= 1 \
            else SyncMultiGPUTrainerParameterServer(nr_gpu)
        launch_train_with_config(config, trainer)
