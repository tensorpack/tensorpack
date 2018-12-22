#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: resnet-dorefa.py

import argparse
import numpy as np
import os
import cv2
import tensorflow as tf

from tensorpack import *
from tensorpack.dataflow import dataset
from tensorpack.tfutils.varreplace import remap_variables

from dorefa import get_dorefa
from imagenet_utils import ImageNetModel, eval_on_ILSVRC12, fbresnet_augmentor

"""
This script loads the pre-trained ResNet-18 model with (W,A,G) = (1,4,32)
It has 59.2% top-1 and 81.5% top-5 validation error on ILSVRC12 validation set.

To run on images:
    ./resnet-dorefa.py --load ResNet-18-14f.npz --run a.jpg b.jpg

To eval on ILSVRC validation set:
    ./resnet-dorefa.py --load ResNet-18-14f.npz --eval --data /path/to/ILSVRC
"""

BITW = 1
BITA = 4
BITG = 32


class Model(ModelDesc):
    def inputs(self):
        return [tf.placeholder(tf.float32, [None, 224, 224, 3], 'input'),
                tf.placeholder(tf.int32, [None], 'label')]

    def build_graph(self, image, label):
        image = image / 256.0

        fw, fa, fg = get_dorefa(BITW, BITA, BITG)

        def new_get_variable(v):
            name = v.op.name
            # don't binarize first and last layer
            if not name.endswith('W') or 'conv1' in name or 'fct' in name:
                return v
            else:
                logger.info("Binarizing weight {}".format(v.op.name))
                return fw(v)

        def nonlin(x):
            return tf.clip_by_value(x, 0.0, 1.0)

        def activate(x):
            return fa(nonlin(x))

        def resblock(x, channel, stride):
            def get_stem_full(x):
                return (LinearWrap(x)
                        .Conv2D('c3x3a', channel, 3)
                        .BatchNorm('stembn')
                        .apply(activate)
                        .Conv2D('c3x3b', channel, 3)())
            channel_mismatch = channel != x.get_shape().as_list()[3]
            if stride != 1 or channel_mismatch or 'pool1' in x.name:
                # handling pool1 is to work around an architecture bug in our model
                if stride != 1 or 'pool1' in x.name:
                    x = AvgPooling('pool', x, stride, stride)
                x = BatchNorm('bn', x)
                x = activate(x)
                shortcut = Conv2D('shortcut', x, channel, 1)
                stem = get_stem_full(x)
            else:
                shortcut = x
                x = BatchNorm('bn', x)
                x = activate(x)
                stem = get_stem_full(x)
            return shortcut + stem

        def group(x, name, channel, nr_block, stride):
            with tf.variable_scope(name + 'blk1'):
                x = resblock(x, channel, stride)
            for i in range(2, nr_block + 1):
                with tf.variable_scope(name + 'blk{}'.format(i)):
                    x = resblock(x, channel, 1)
            return x

        with remap_variables(new_get_variable), \
                argscope(BatchNorm, decay=0.9, epsilon=1e-4), \
                argscope(Conv2D, use_bias=False, nl=tf.identity):
            logits = (LinearWrap(image)
                      # use explicit padding here, because our private training framework has
                      # different padding mechanisms from TensorFlow
                      .tf.pad([[0, 0], [3, 2], [3, 2], [0, 0]])
                      .Conv2D('conv1', 64, 7, stride=2, padding='VALID', use_bias=True)
                      .tf.pad([[0, 0], [1, 1], [1, 1], [0, 0]], 'SYMMETRIC')
                      .MaxPooling('pool1', 3, 2, padding='VALID')
                      .apply(group, 'conv2', 64, 2, 1)
                      .apply(group, 'conv3', 128, 2, 2)
                      .apply(group, 'conv4', 256, 2, 2)
                      .apply(group, 'conv5', 512, 2, 2)
                      .BatchNorm('lastbn')
                      .apply(nonlin)
                      .GlobalAvgPooling('gap')
                      .tf.multiply(49)  # this is due to a bug in our model design
                      .FullyConnected('fct', 1000)())
        tf.nn.softmax(logits, name='output')
        ImageNetModel.compute_loss_and_error(logits, label)


def get_inference_augmentor():
    return fbresnet_augmentor(False)


def run_image(model, sess_init, inputs):
    pred_config = PredictConfig(
        model=model,
        session_init=sess_init,
        input_names=['input'],
        output_names=['output']
    )
    predict_func = OfflinePredictor(pred_config)
    meta = dataset.ILSVRCMeta()
    words = meta.get_synset_words_1000()

    transformers = get_inference_augmentor()
    for f in inputs:
        assert os.path.isfile(f)
        img = cv2.imread(f).astype('float32')
        assert img is not None

        img = transformers.augment(img)[np.newaxis, :, :, :]
        o = predict_func(img)
        prob = o[0][0]
        ret = prob.argsort()[-10:][::-1]

        names = [words[i] for i in ret]
        print(f + ":")
        print(list(zip(names, prob[ret])))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='the physical ids of GPUs to use')
    parser.add_argument('--load', help='load a npz pretrained model')
    parser.add_argument('--data', help='ILSVRC dataset dir')
    parser.add_argument('--dorefa',
                        help='number of bits for W,A,G, separated by comma. Defaults to \'1,4,32\'',
                        default='1,4,32')
    parser.add_argument(
        '--run', help='run on a list of images with the pretrained model', nargs='*')
    parser.add_argument('--eval', action='store_true')
    args = parser.parse_args()

    BITW, BITA, BITG = map(int, args.dorefa.split(','))

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if args.eval:
        ds = dataset.ILSVRC12(args.data, 'val', shuffle=False)
        ds = AugmentImageComponent(ds, get_inference_augmentor())
        ds = BatchData(ds, 192, remainder=True)
        eval_on_ILSVRC12(Model(), get_model_loader(args.load), ds)
    elif args.run:
        assert args.load.endswith('.npz')
        run_image(Model(), DictRestore(dict(np.load(args.load))), args.run)
