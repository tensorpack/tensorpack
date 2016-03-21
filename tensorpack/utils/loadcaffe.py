#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: loadcaffe.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

from collections import namedtuple, defaultdict
from abc import abstractmethod
import os

from six.moves import zip

from .utils import change_env
from . import logger

def get_processor():
    ret = {}
    def process_conv(layer_name, param):
        assert len(param) == 2
        # caffe: ch_out, ch_in, h, w
        return {layer_name + '/W': param[0].data.transpose(2,3,1,0),
                layer_name + '/b': param[1].data}
    ret['Convolution'] = process_conv

    # XXX caffe has an 'transpose' option for fc/W
    def process_fc(layer_name, param):
        assert len(param) == 2
        return {layer_name + '/W': param[0].data.transpose(),
                layer_name + '/b': param[1].data}
    ret['InnerProduct'] = process_fc
    return ret

def load_caffe(model_desc, model_file):
    """
    return a dict of params
    """
    param_dict = {}
    param_processors = get_processor()

    with change_env('GLOG_minloglevel', '2'):
        import caffe
        net = caffe.Net(model_desc, model_file, caffe.TEST)
    layer_names = net._layer_names
    for layername, layer in zip(layer_names, net.layers):
        if layer.type in param_processors:
            param_dict.update(param_processors[layer.type](layername, layer.blobs))
        else:
            assert len(layer.blobs) == 0, len(layer.blobs)
    logger.info("Model loaded from caffe. Params: " + \
                " ".join(sorted(param_dict.keys())))
    return param_dict

if __name__ == '__main__':
    ret = load_caffe('/home/wyx/Work/DL/caffe/models/VGG/VGG_ILSVRC_16_layers_deploy.prototxt',
               '/home/wyx/Work/DL/caffe/models/VGG/VGG_ILSVRC_16_layers.caffemodel')
