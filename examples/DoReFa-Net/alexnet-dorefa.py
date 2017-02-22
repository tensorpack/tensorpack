#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: alexnet-dorefa.py
# Author: Yuxin Wu, Yuheng Zou ({wyx,zyh}@megvii.com)

import cv2
import tensorflow as tf
import argparse
import numpy as np
import multiprocessing
import msgpack
import os
import sys

from tensorpack import *
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.tfutils.summary import *
from tensorpack.tfutils.varreplace import remap_get_variable
from dorefa import get_dorefa

"""
This is a tensorpack script for the ImageNet results in paper:
DoReFa-Net: Training Low Bitwidth Convolutional Neural Networks with Low Bitwidth Gradients
http://arxiv.org/abs/1606.06160

The original experiements are performed on a proprietary framework.
This is our attempt to reproduce it on tensorpack & TensorFlow.

Accuracy:
    Trained with 4 GPUs and (W,A,G)=(1,2,6), it can reach top-1 single-crop validation error of 51%,
    after 70 epochs. This number is a bit better than what's in the paper
    probably due to more sophisticated augmentors.

    Note that the effective batch size in SyncMultiGPUTrainer is actually
    BATCH_SIZE * NUM_GPU. With a different number of GPUs in use, things might
    be a bit different, especially for learning rate.

    With (W,A,G)=(32,32,32) -- full precision baseline, 43% error.
    With (W,A,G)=(1,32,32) -- BWN, 46% error.
    With (W,A,G)=(1,2,6), 51% error.
    With (W,A,G)=(1,2,4), 63% error.

Speed:
    About 2.2 iteration/s on 1 TitanX. (Each epoch is set to 10000 iterations)
    Note that this code was written early without using NCHW format. You
    should expect a 30% speed up after switching to NCHW format.

To Train, for example:
    ./alexnet-dorefa.py --dorefa 1,2,6 --data PATH --gpu 0,1

    PATH should look like:
    PATH/
      train/
        n02134418/
          n02134418_198.JPEG
          ...
        ...
      val/
        ILSVRC2012_val_00000001.JPEG
        ...

    And you'll need the following to be able to fetch data efficiently
        Fast disk random access (Not necessarily SSD. I used a RAID of HDD, but not sure if plain HDD is enough)
        More than 12 CPU cores (for data processing)
        More than 10G of free memory

To Run Pretrained Model:
    ./alexnet-dorefa.py --load alexnet-126.npy --run a.jpg --dorefa 1,2,6
"""

BITW = 1
BITA = 2
BITG = 6
TOTAL_BATCH_SIZE = 128
BATCH_SIZE = None


class Model(ModelDesc):
    def _get_inputs(self):
        return [InputDesc(tf.float32, [None, 224, 224, 3], 'input'),
                InputDesc(tf.int32, [None], 'label')]

    def _build_graph(self, inputs):
        image, label = inputs
        image = image / 255.0

        fw, fa, fg = get_dorefa(BITW, BITA, BITG)

        old_get_variable = tf.get_variable

        # monkey-patch tf.get_variable to apply fw
        def new_get_variable(v):
            name = v.op.name
            # don't binarize first and last layer
            if not name.endswith('W') or 'conv0' in name or 'fct' in name:
                return v
            else:
                logger.info("Binarizing weight {}".format(v.op.name))
                return fw(v)

        def nonlin(x):
            if BITA == 32:
                return tf.nn.relu(x)    # still use relu for 32bit cases
            return tf.clip_by_value(x, 0.0, 1.0)

        def activate(x):
            return fa(nonlin(x))

        with remap_get_variable(new_get_variable), \
                argscope(BatchNorm, decay=0.9, epsilon=1e-4), \
                argscope([Conv2D, FullyConnected], use_bias=False, nl=tf.identity):
            logits = (LinearWrap(image)
                      .Conv2D('conv0', 96, 12, stride=4, padding='VALID')
                      .apply(activate)
                      .Conv2D('conv1', 256, 5, padding='SAME', split=2)
                      .apply(fg)
                      .BatchNorm('bn1')
                      .MaxPooling('pool1', 3, 2, padding='SAME')
                      .apply(activate)

                      .Conv2D('conv2', 384, 3)
                      .apply(fg)
                      .BatchNorm('bn2')
                      .MaxPooling('pool2', 3, 2, padding='SAME')
                      .apply(activate)

                      .Conv2D('conv3', 384, 3, split=2)
                      .apply(fg)
                      .BatchNorm('bn3')
                      .apply(activate)

                      .Conv2D('conv4', 256, 3, split=2)
                      .apply(fg)
                      .BatchNorm('bn4')
                      .MaxPooling('pool4', 3, 2, padding='VALID')
                      .apply(activate)

                      .FullyConnected('fc0', 4096)
                      .apply(fg)
                      .BatchNorm('bnfc0')
                      .apply(activate)

                      .FullyConnected('fc1', 4096)
                      .apply(fg)
                      .BatchNorm('bnfc1')
                      .apply(nonlin)
                      .FullyConnected('fct', 1000, use_bias=True)())

        prob = tf.nn.softmax(logits, name='output')

        cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label)
        cost = tf.reduce_mean(cost, name='cross_entropy_loss')

        wrong = prediction_incorrect(logits, label, 1, name='wrong-top1')
        add_moving_summary(tf.reduce_mean(wrong, name='train-error-top1'))
        wrong = prediction_incorrect(logits, label, 5, name='wrong-top5')
        add_moving_summary(tf.reduce_mean(wrong, name='train-error-top5'))

        # weight decay on all W of fc layers
        wd_cost = regularize_cost('fc.*/W', l2_regularizer(5e-6), name='regularize_cost')

        add_param_summary(('.*/W', ['histogram', 'rms']))
        self.cost = tf.add_n([cost, wd_cost], name='cost')
        add_moving_summary(cost, wd_cost, self.cost)

    def _get_optimizer(self):
        lr = get_scalar_var('learning_rate', 1e-4, summary=True)
        return tf.train.AdamOptimizer(lr, epsilon=1e-5)


def get_data(dataset_name):
    isTrain = dataset_name == 'train'
    ds = dataset.ILSVRC12(args.data, dataset_name, shuffle=isTrain)

    meta = dataset.ILSVRCMeta()
    pp_mean = meta.get_per_pixel_mean()
    pp_mean_224 = pp_mean[16:-16, 16:-16, :]

    if isTrain:
        class Resize(imgaug.ImageAugmentor):
            def __init__(self):
                self._init(locals())

            def _augment(self, img, _):
                h, w = img.shape[:2]
                size = 224
                scale = self.rng.randint(size, 308) * 1.0 / min(h, w)
                scaleX = scale * self.rng.uniform(0.85, 1.15)
                scaleY = scale * self.rng.uniform(0.85, 1.15)
                desSize = map(int, (max(size, min(w, scaleX * w)),
                                    max(size, min(h, scaleY * h))))
                dst = cv2.resize(img, tuple(desSize),
                                 interpolation=cv2.INTER_CUBIC)
                return dst

        augmentors = [
            Resize(),
            imgaug.Rotation(max_deg=10),
            imgaug.RandomApplyAug(imgaug.GaussianBlur(3), 0.5),
            imgaug.Brightness(30, True),
            imgaug.Gamma(),
            imgaug.Contrast((0.8, 1.2), True),
            imgaug.RandomCrop((224, 224)),
            imgaug.RandomApplyAug(imgaug.JpegNoise(), 0.8),
            imgaug.RandomApplyAug(imgaug.GaussianDeform(
                [(0.2, 0.2), (0.2, 0.8), (0.8, 0.8), (0.8, 0.2)],
                (224, 224), 0.2, 3), 0.1),
            imgaug.Flip(horiz=True),
            imgaug.MapImage(lambda x: x - pp_mean_224),
        ]
    else:
        def resize_func(im):
            h, w = im.shape[:2]
            scale = 256.0 / min(h, w)
            desSize = map(int, (max(224, min(w, scale * w)),
                                max(224, min(h, scale * h))))
            im = cv2.resize(im, tuple(desSize), interpolation=cv2.INTER_CUBIC)
            return im
        augmentors = [
            imgaug.MapImage(resize_func),
            imgaug.CenterCrop((224, 224)),
            imgaug.MapImage(lambda x: x - pp_mean_224),
        ]
    ds = AugmentImageComponent(ds, augmentors)
    ds = BatchData(ds, BATCH_SIZE, remainder=not isTrain)
    if isTrain:
        ds = PrefetchDataZMQ(ds, min(12, multiprocessing.cpu_count()))
    return ds


def get_config():
    logger.auto_set_dir()
    data_train = get_data('train')
    data_test = get_data('val')

    return TrainConfig(
        dataflow=data_train,
        callbacks=[
            ModelSaver(),
            # HumanHyperParamSetter('learning_rate'),
            ScheduledHyperParamSetter(
                'learning_rate', [(56, 2e-5), (64, 4e-6)]),
            InferenceRunner(data_test,
                            [ScalarStats('cost'),
                             ClassificationError('wrong-top1', 'val-error-top1'),
                             ClassificationError('wrong-top5', 'val-error-top5')])
        ],
        model=Model(),
        steps_per_epoch=10000,
        max_epoch=100,
    )


def run_image(model, sess_init, inputs):
    pred_config = PredictConfig(
        model=model,
        session_init=sess_init,
        input_names=['input'],
        output_names=['output']
    )
    predictor = OfflinePredictor(pred_config)
    meta = dataset.ILSVRCMeta()
    pp_mean = meta.get_per_pixel_mean()
    pp_mean_224 = pp_mean[16:-16, 16:-16, :]
    words = meta.get_synset_words_1000()

    def resize_func(im):
        h, w = im.shape[:2]
        scale = 256.0 / min(h, w)
        desSize = map(int, (max(224, min(w, scale * w)),
                            max(224, min(h, scale * h))))
        im = cv2.resize(im, tuple(desSize), interpolation=cv2.INTER_CUBIC)
        return im
    transformers = imgaug.AugmentorList([
        imgaug.MapImage(resize_func),
        imgaug.CenterCrop((224, 224)),
        imgaug.MapImage(lambda x: x - pp_mean_224),
    ])
    for f in inputs:
        assert os.path.isfile(f)
        img = cv2.imread(f).astype('float32')
        assert img is not None

        img = transformers.augment(img)[np.newaxis, :, :, :]
        outputs = predictor([img])[0]
        prob = outputs[0]
        ret = prob.argsort()[-10:][::-1]

        names = [words[i] for i in ret]
        print(f + ":")
        print(list(zip(names, prob[ret])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='the physical ids of GPUs to use')
    parser.add_argument('--load', help='load a checkpoint, or a npy (given as the pretrained model)')
    parser.add_argument('--data', help='ILSVRC dataset dir')
    parser.add_argument('--dorefa',
                        help='number of bits for W,A,G, separated by comma', required=True)
    parser.add_argument('--run', help='run on a list of images with the pretrained model', nargs='*')
    args = parser.parse_args()

    BITW, BITA, BITG = map(int, args.dorefa.split(','))

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if args.run:
        assert args.load.endswith('.npy')
        run_image(Model(), ParamRestore(np.load(args.load, encoding='latin1').item()), args.run)
        sys.exit()

    assert args.gpu is not None, "Need to specify a list of gpu for training!"
    NR_GPU = len(args.gpu.split(','))
    BATCH_SIZE = TOTAL_BATCH_SIZE // NR_GPU
    logger.info("Batch per tower: {}".format(BATCH_SIZE))

    config = get_config()
    if args.load:
        config.session_init = SaverRestore(args.load)
    if args.gpu:
        config.nr_tower = len(args.gpu.split(','))
    SyncMultiGPUTrainer(config).train()
