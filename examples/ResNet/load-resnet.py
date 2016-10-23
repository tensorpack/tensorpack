#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: load-ResNet.py
# Author: Eric Yujia Huang yujiah1@andrew.cmu.edu
#         Yuxin Wu <ppwwyyxx@gmail.com>
# 

import cv2
import tensorflow as tf
import argparse
import numpy as np
from six.moves import zip
from tensorflow.contrib.layers import variance_scaling_initializer

from tensorpack import *
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.tfutils.summary import *
from tensorpack.dataflow.dataset import ILSVRCMeta

"""
Usage:
    python2 -m tensorpack.utils.loadcaffe PATH/TO/CAFFE/{ResNet-101-deploy.prototxt,ResNet-101-model.caffemodel} ResNet101.npy
    ./load-alexnet.py --load ResNet-101.npy --input cat.png --depth 101
"""
MODEL_DEPTH = None

class Model(ModelDesc):

    def _get_input_vars(self):
        return [InputVar(tf.float32, [None, 224, 224, 3], 'input'),
		InputVar(tf.int32, [None],'label')]

    def _build_graph(self, input_vars):
        image, label = input_vars

        def caffe_shortcut(l, n_in, n_out, stride):
            if n_in != n_out:
                l = Conv2D('convshortcut', l, n_out, 1, stride=stride)
                return BatchNorm('bnshortcut', l)
            else:
                return l

        def caffe_bottleneck(l, ch_out, stride, preact):
            ch_in = l.get_shape().as_list()[-1]
            input = l
            if preact == 'both_preact':
                l = tf.nn.relu(l, name='preact-relu')
                input = l
            l = Conv2D('conv1', l, ch_out, 1)
            l = BatchNorm('bn1', l)
            l = tf.nn.relu(l)
            l = Conv2D('conv2', l, ch_out, 3, stride=stride)
            l = BatchNorm('bn2', l)
            l = tf.nn.relu(l)
            l = Conv2D('conv3', l, ch_out * 4, 1)
            l = BatchNorm('bn3', l)  # put bn at the bottom
            return l + caffe_shortcut(input, ch_in, ch_out * 4, stride)

        def layer(l, layername, block_func, features, count, stride, first=False):
            with tf.variable_scope(layername):
                with tf.variable_scope('block0'):
                    l = block_func(l, features, stride,
                            'no_preact' if first else 'both_preact')
                for i in range(1, count):
                    with tf.variable_scope('block{}'.format(i)):
                        l = block_func(l, features, 1, 'both_preact')
                return l

        cfg = {
            50: ([3,4,6,3], caffe_bottleneck),
            101: ([3,4,23,3], caffe_bottleneck),
            152: ([3,8,36,3], caffe_bottleneck)
        }

        defs, block_func = cfg[MODEL_DEPTH]

        with argscope(Conv2D, nl=tf.identity, use_bias=False,
                W_init=variance_scaling_initializer(mode='FAN_OUT')):
            fc1000l = (LinearWrap(image)
                .Conv2D('conv0', 64, 7, stride=2, nl=BNReLU ) 
                .MaxPooling('pool0', shape=3, stride=2, padding='SAME')
                .apply(layer, 'group0', block_func, 64, defs[0], 1, first=True)
                .apply(layer, 'group1', block_func, 128, defs[1], 2)
                .apply(layer, 'group2', block_func, 256, defs[2], 2)
                .apply(layer, 'group3', block_func, 512, defs[3], 2)
                .tf.nn.relu()
                .GlobalAvgPooling('gap')
                .FullyConnected('fc1000', 1000, nl=tf.identity)())

            prob = tf.nn.softmax(fc1000l, name='prob_output')
       


def run_test(path, input):
    image_mean = np.array([0.485, 0.456, 0.406], dtype='float32')
    param = np.load(path).item()

    resNet_param = { caffeResNet2tensorpackResNet(k) :v for k, v in param.iteritems()}
    pred_config = PredictConfig(
        model=Model(),
        input_var_names=['input'],
        session_init=ParamRestore(resNet_param),
        session_config=get_default_sess_config(0.9),
        output_var_names=['prob_output']   # output:0 is the probability distribution
    )
    predict_func = get_predict_func(pred_config)

    remap_func = lambda  x: (x  - image_mean * 255)
    im = cv2.imread(input)
    im = remap_func(cv2.resize(im, (224,224)))
    im = np.reshape( im, (1, 224, 224, 3)).astype('float32')
    input = [im]
    prob = predict_func(input)[0]

    ret = prob[0].argsort()[-10:][::-1]
    print(ret)

    meta = ILSVRCMeta().get_synset_words_1000()
    print([meta[k] for k in ret])


def caffeResNet2tensorpackResNet(caffe_layer_name):
    import re
    map = dict()
        #begining & ending stage
    map['conv1/W'] = 'conv0/W'
    map['conv1/b'] = 'conv0/b'
    map['bn_conv1/beta'] = 'conv0/bn/beta'
    map['bn_conv1/gamma'] = 'conv0/bn/gamma'
    map['bn_conv1/mean/EMA'] = 'conv0/bn/mean/EMA'
    map['bn_conv1/variance/EMA'] = 'conv0/bn/variance/EMA'
    map['fc1000/W'] = 'fc1000/W'
    map['fc1000/b'] = 'fc1000/b'
    if map.get(caffe_layer_name) != None:
        print(caffe_layer_name + ' --> ' + map[caffe_layer_name])
        return map[caffe_layer_name]

    print(caffe_layer_name)

    layer_id = None
    layer_type = None
    layer_block = None
    layer_branch = None
    layer_group = None
    s = re.search('([a-z]*)([0-9]*)([a-z]*)_branch([0-9])([a-z])', caffe_layer_name, re.IGNORECASE)
    if s == None:
        s = re.search('([a-z]*)([0-9]*)([a-z]*)_branch([0-9])', caffe_layer_name, re.IGNORECASE)
    else:
        layer_id = s.group(5)

    if s.group(0) == caffe_layer_name[0:caffe_layer_name.index('/')]:
        layer_type = s.group(1)
        layer_group = s.group(2)
        layer_block = ord(s.group(3)) - ord('a') 
        layer_branch = s.group(4)
    else:
        # print('s group ' + s.group(0))
        s = re.search('([a-z]*)([0-9]*)([a-z]*)([0-9]*)_branch([0-9])([a-z])', caffe_layer_name, re.IGNORECASE)
        if s == None:
            s = re.search('([a-z]*)([0-9]*)([a-z]*)([0-9]*)_branch([0-9])', caffe_layer_name, re.IGNORECASE)
        else:
            layer_id = s.group(6)

        layer_type = s.group(1)
        layer_group = s.group(2)
        layer_block_part1 = s.group(3)
        layer_block_part2 = s.group(4)
        if layer_block_part1 == 'a':
            layer_block = 0
        elif layer_block_part1 == 'b':
            layer_block = int(layer_block_part2)
        else:
            print('model block error!')

        layer_branch = s.group(5)  

    if s.group(0) != caffe_layer_name[0:caffe_layer_name.index('/')]:
        print('model depth error!')
        # error handling

    
    type_dict = {'res': '/conv', 'bn':'/bn', 'scale':'/bn'}
    shortcut_dict = {'res': '/convshortcut', 'bn':'/bnshortcut', 'scale':'/bnshortcut'}

    tf_name = caffe_layer_name[caffe_layer_name.index('/'):]

    if layer_branch == '2':
        tf_name = 'group' + str( int(layer_group) - int('2') ) + \
              '/block' + str( layer_block ) + \
              type_dict[layer_type] + str( ord(layer_id) - ord('a') + 1) + tf_name
    elif layer_branch == '1':
        tf_name = 'group' + str( int(layer_group) - int('2') ) + \
              '/block' + str(layer_block) + \
              shortcut_dict[layer_type] + tf_name
    else:
        print('renaming error!')
        # error handling
    print(caffe_layer_name + ' --> ' + tf_name)
    return tf_name

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.') # nargs='*' in multi mode
    parser.add_argument('--load',
                        help='.npy model file generated by tensorpack.utils.loadcaffe',
                        required=True)
    parser.add_argument('--input', help='an input image', required=True)
    parser.add_argument('--depth', help='resnet depth', required=True, type=int, choices=[50, 101, 152])

    args = parser.parse_args()
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    # run resNet with given model (in npy format)
    MODEL_DEPTH = args.depth
    run_test(args.load, args.input)