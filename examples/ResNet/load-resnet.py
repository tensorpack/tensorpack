#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: load-resnet.py
# Author: Eric Yujia Huang yujiah1@andrew.cmu.edu
#         Yuxin Wu <ppwwyyxx@gmail.com>

import cv2
import tensorflow as tf
import argparse
import os
import re
import numpy as np
import six
from six.moves import zip
from tensorflow.contrib.layers import variance_scaling_initializer

from tensorpack import *
from tensorpack.utils import logger
from tensorpack.utils.stats import RatioCounter
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.tfutils.summary import *
from tensorpack.dataflow.dataset import ILSVRCMeta

MODEL_DEPTH = None


class Model(ModelDesc):
    def _get_inputs(self):
        return [InputDesc(tf.float32, [None, 224, 224, 3], 'input'),
                InputDesc(tf.int32, [None], 'label')]

    def _build_graph(self, input_vars):
        image, label = input_vars

        def shortcut(l, n_in, n_out, stride):
            if n_in != n_out:
                l = Conv2D('convshortcut', l, n_out, 1, stride=stride)
                return BatchNorm('bnshortcut', l)
            else:
                return l

        def bottleneck(l, ch_out, stride, preact):
            ch_in = l.get_shape().as_list()[-1]
            input = l
            if preact == 'both_preact':
                l = tf.nn.relu(l, name='preact-relu')
                input = l
            l = Conv2D('conv1', l, ch_out, 1, stride=stride)
            l = BatchNorm('bn1', l)
            l = tf.nn.relu(l)
            l = Conv2D('conv2', l, ch_out, 3)
            l = BatchNorm('bn2', l)
            l = tf.nn.relu(l)
            l = Conv2D('conv3', l, ch_out * 4, 1)
            l = BatchNorm('bn3', l)  # put bn at the bottom
            return l + shortcut(input, ch_in, ch_out * 4, stride)

        def layer(l, layername, features, count, stride, first=False):
            with tf.variable_scope(layername):
                with tf.variable_scope('block0'):
                    l = bottleneck(l, features, stride,
                                   'no_preact' if first else 'both_preact')
                for i in range(1, count):
                    with tf.variable_scope('block{}'.format(i)):
                        l = bottleneck(l, features, 1, 'both_preact')
                return l

        cfg = {
            50: ([3, 4, 6, 3]),
            101: ([3, 4, 23, 3]),
            152: ([3, 8, 36, 3])
        }
        defs = cfg[MODEL_DEPTH]

        with argscope(Conv2D, nl=tf.identity, use_bias=False,
                      W_init=variance_scaling_initializer(mode='FAN_OUT')):
            # tensorflow with padding=SAME will by default pad [2,3] here.
            # but caffe conv with stride will pad [3,3]
            image = tf.pad(image, [[0, 0], [3, 3], [3, 3], [0, 0]])
            fc1000 = (LinearWrap(image)
                      .Conv2D('conv0', 64, 7, stride=2, nl=BNReLU, padding='VALID')
                      .MaxPooling('pool0', shape=3, stride=2, padding='SAME')
                      .apply(layer, 'group0', 64, defs[0], 1, first=True)
                      .apply(layer, 'group1', 128, defs[1], 2)
                      .apply(layer, 'group2', 256, defs[2], 2)
                      .apply(layer, 'group3', 512, defs[3], 2)
                      .tf.nn.relu()
                      .GlobalAvgPooling('gap')
                      .FullyConnected('fc1000', 1000, nl=tf.identity)())
        prob = tf.nn.softmax(fc1000, name='prob')
        nr_wrong = prediction_incorrect(fc1000, label, name='wrong-top1')
        nr_wrong = prediction_incorrect(fc1000, label, 5, name='wrong-top5')


def get_inference_augmentor():
    # load ResNet mean from Kaiming:
    # from tensorpack.utils.loadcaffe import get_caffe_pb
    # obj = get_caffe_pb().BlobProto()
    # obj.ParseFromString(open('ResNet_mean.binaryproto').read())
    # pp_mean_224 = np.array(obj.data).reshape(3, 224, 224).transpose(1,2,0)

    meta = ILSVRCMeta()
    pp_mean = meta.get_per_pixel_mean()
    pp_mean_224 = pp_mean[16:-16, 16:-16, :]

    transformers = imgaug.AugmentorList([
        imgaug.ResizeShortestEdge(256),
        imgaug.CenterCrop((224, 224)),
        imgaug.MapImage(lambda x: x - pp_mean_224),
    ])
    return transformers


def run_test(params, input):
    pred_config = PredictConfig(
        model=Model(),
        session_init=ParamRestore(params),
        input_names=['input'],
        output_names=['prob']
    )
    predict_func = OfflinePredictor(pred_config)

    prepro = get_inference_augmentor()
    im = cv2.imread(input).astype('float32')
    im = prepro.augment(im)
    im = np.reshape(im, (1, 224, 224, 3))
    outputs = predict_func([im])
    prob = outputs[0]

    ret = prob[0].argsort()[-10:][::-1]
    print(ret)
    meta = ILSVRCMeta().get_synset_words_1000()
    print([meta[k] for k in ret])


def eval_on_ILSVRC12(params, data_dir):
    ds = dataset.ILSVRC12(data_dir, 'val', shuffle=False, dir_structure='train')
    ds = AugmentImageComponent(ds, get_inference_augmentor())
    ds = BatchData(ds, 128, remainder=True)
    pred_config = PredictConfig(
        model=Model(),
        session_init=ParamRestore(params),
        input_names=['input', 'label'],
        output_names=['wrong-top1', 'wrong-top5']
    )
    pred = SimpleDatasetPredictor(pred_config, ds)
    acc1, acc5 = RatioCounter(), RatioCounter()
    for o in pred.get_result():
        batch_size = o[0].shape[0]
        acc1.feed(o[0].sum(), batch_size)
        acc5.feed(o[1].sum(), batch_size)
    print("Top1 Error: {}".format(acc1.ratio))
    print("Top5 Error: {}".format(acc5.ratio))


def name_conversion(caffe_layer_name):
    """ Convert a caffe parameter name to a tensorflow parameter name as
        defined in the above model """
    # beginning & end mapping
    NAME_MAP = {'bn_conv1/beta': 'conv0/bn/beta',
                'bn_conv1/gamma': 'conv0/bn/gamma',
                'bn_conv1/mean/EMA': 'conv0/bn/mean/EMA',
                'bn_conv1/variance/EMA': 'conv0/bn/variance/EMA',
                'conv1/W': 'conv0/W', 'conv1/b': 'conv0/b',
                'fc1000/W': 'fc1000/W', 'fc1000/b': 'fc1000/b'}
    if caffe_layer_name in NAME_MAP:
        return NAME_MAP[caffe_layer_name]

    s = re.search('([a-z]+)([0-9]+)([a-z]+)_', caffe_layer_name)
    if s is None:
        s = re.search('([a-z]+)([0-9]+)([a-z]+)([0-9]+)_', caffe_layer_name)
        layer_block_part1 = s.group(3)
        layer_block_part2 = s.group(4)
        assert layer_block_part1 in ['a', 'b']
        layer_block = 0 if layer_block_part1 == 'a' else int(layer_block_part2)
    else:
        layer_block = ord(s.group(3)) - ord('a')
    layer_type = s.group(1)
    layer_group = s.group(2)

    layer_branch = int(re.search('_branch([0-9])', caffe_layer_name).group(1))
    assert layer_branch in [1, 2]
    if layer_branch == 2:
        layer_id = re.search('_branch[0-9]([a-z])/', caffe_layer_name).group(1)
        layer_id = ord(layer_id) - ord('a') + 1

    TYPE_DICT = {'res': 'conv', 'bn': 'bn'}

    tf_name = caffe_layer_name[caffe_layer_name.index('/'):]
    layer_type = TYPE_DICT[layer_type] + \
        (str(layer_id) if layer_branch == 2 else 'shortcut')
    tf_name = 'group{}/block{}/{}'.format(
        int(layer_group) - 2, layer_block, layer_type) + tf_name
    return tf_name


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--load', required=True,
                        help='.npy model file generated by tensorpack.utils.loadcaffe')
    parser.add_argument('-d', '--depth', help='resnet depth', required=True, type=int, choices=[50, 101, 152])
    parser.add_argument('--input', help='an input image')
    parser.add_argument('--eval', help='ILSVRC dir to run validation on')

    args = parser.parse_args()
    assert args.input or args.eval, "Choose either input or eval!"
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    MODEL_DEPTH = args.depth

    param = np.load(args.load, encoding='latin1').item()
    resnet_param = {}
    for k, v in six.iteritems(param):
        try:
            newname = name_conversion(k)
        except:
            logger.error("Exception when processing caffe layer {}".format(k))
            raise
        logger.info("Name Transform: " + k + ' --> ' + newname)
        resnet_param[newname] = v

    if args.eval:
        eval_on_ILSVRC12(resnet_param, args.eval)
    else:
        run_test(resnet_param, args.input)
