#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: load-resnet.py
# Author: Eric Yujia Huang <yujiah1@andrew.cmu.edu>
#         Yuxin Wu

import argparse
import functools
import numpy as np
import re
import cv2
import six
import tensorflow as tf

from tensorpack import *
from tensorpack.dataflow.dataset import ILSVRCMeta
from tensorpack.utils import logger

from imagenet_utils import ImageNetModel, eval_on_ILSVRC12, get_imagenet_dataflow
from resnet_model import resnet_bottleneck, resnet_group

DEPTH = None
CFG = {
    50: ([3, 4, 6, 3]),
    101: ([3, 4, 23, 3]),
    152: ([3, 8, 36, 3])
}


class Model(ModelDesc):
    def inputs(self):
        return [tf.placeholder(tf.float32, [None, 224, 224, 3], 'input'),
                tf.placeholder(tf.int32, [None], 'label')]

    def build_graph(self, image, label):
        blocks = CFG[DEPTH]

        bottleneck = functools.partial(resnet_bottleneck, stride_first=True)

        # tensorflow with padding=SAME will by default pad [2,3] here.
        # but caffe conv with stride will pad [3,2]
        image = tf.pad(image, [[0, 0], [3, 2], [3, 2], [0, 0]])
        image = tf.transpose(image, [0, 3, 1, 2])
        with argscope([Conv2D, MaxPooling, GlobalAvgPooling, BatchNorm],
                      data_format='channels_first'), \
                argscope(Conv2D, use_bias=False):
            logits = (LinearWrap(image)
                      .Conv2D('conv0', 64, 7, strides=2, activation=BNReLU, padding='VALID')
                      .MaxPooling('pool0', 3, strides=2, padding='SAME')
                      .apply2(resnet_group, 'group0', bottleneck, 64, blocks[0], 1)
                      .apply2(resnet_group, 'group1', bottleneck, 128, blocks[1], 2)
                      .apply2(resnet_group, 'group2', bottleneck, 256, blocks[2], 2)
                      .apply2(resnet_group, 'group3', bottleneck, 512, blocks[3], 2)
                      .GlobalAvgPooling('gap')
                      .FullyConnected('linear', 1000)())
        tf.nn.softmax(logits, name='prob')
        ImageNetModel.compute_loss_and_error(logits, label)


def get_inference_augmentor():
    # load ResNet mean from Kaiming:
    # from tensorpack.utils.loadcaffe import get_caffe_pb
    # obj = get_caffe_pb().BlobProto()
    # obj.ParseFromString(open('ResNet_mean.binaryproto').read())
    # pp_mean_224 = np.array(obj.data).reshape(3, 224, 224).transpose(1,2,0)

    meta = ILSVRCMeta()
    pp_mean = meta.get_per_pixel_mean()
    pp_mean_224 = pp_mean[16:-16, 16:-16, :]

    transformers = [
        imgaug.ResizeShortestEdge(256),
        imgaug.CenterCrop((224, 224)),
        imgaug.MapImage(lambda x: x - pp_mean_224),
    ]
    return transformers


def run_test(params, input):
    pred_config = PredictConfig(
        model=Model(),
        session_init=DictRestore(params),
        input_names=['input'],
        output_names=['prob']
    )
    predict_func = OfflinePredictor(pred_config)

    prepro = imgaug.AugmentorList(get_inference_augmentor())
    im = cv2.imread(input).astype('float32')
    im = prepro.augment(im)
    im = np.reshape(im, (1, 224, 224, 3))
    outputs = predict_func(im)
    prob = outputs[0]

    ret = prob[0].argsort()[-10:][::-1]
    print(ret)
    meta = ILSVRCMeta().get_synset_words_1000()
    print([meta[k] for k in ret])


def name_conversion(caffe_layer_name):
    """ Convert a caffe parameter name to a tensorflow parameter name as
        defined in the above model """
    # beginning & end mapping
    NAME_MAP = {'bn_conv1/beta': 'conv0/bn/beta',
                'bn_conv1/gamma': 'conv0/bn/gamma',
                'bn_conv1/mean/EMA': 'conv0/bn/mean/EMA',
                'bn_conv1/variance/EMA': 'conv0/bn/variance/EMA',
                'conv1/W': 'conv0/W', 'conv1/b': 'conv0/b',
                'fc1000/W': 'linear/W', 'fc1000/b': 'linear/b'}
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

    TYPE_DICT = {'res': 'conv{}', 'bn': 'conv{}/bn'}
    layer_type = TYPE_DICT[layer_type].format(layer_id if layer_branch == 2 else 'shortcut')

    tf_name = caffe_layer_name[caffe_layer_name.index('/'):]
    tf_name = 'group{}/block{}/{}'.format(
        int(layer_group) - 2, layer_block, layer_type) + tf_name
    return tf_name


def convert_param_name(param):
    resnet_param = {}
    for k, v in six.iteritems(param):
        try:
            newname = name_conversion(k)
        except Exception:
            logger.error("Exception when processing caffe layer {}".format(k))
            raise
        logger.info("Name Transform: " + k + ' --> ' + newname)
        resnet_param[newname] = v
    return resnet_param


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', required=True,
                        help='.npz model file generated by tensorpack.utils.loadcaffe')
    parser.add_argument('-d', '--depth', help='resnet depth', required=True, type=int, choices=[50, 101, 152])
    parser.add_argument('--input', help='an input image')
    parser.add_argument('--convert', help='npz output file to save the converted model')
    parser.add_argument('--eval', help='ILSVRC dir to run validation on')

    args = parser.parse_args()
    DEPTH = args.depth

    param = dict(np.load(args.load))
    param = convert_param_name(param)

    if args.convert:
        assert args.convert.endswith('.npz')
        np.savez_compressed(args.convert, **param)

    if args.eval:
        ds = get_imagenet_dataflow(args.eval, 'val', 128, get_inference_augmentor())
        eval_on_ILSVRC12(Model(), DictRestore(param), ds)
    elif args.input:
        run_test(param, args.input)
