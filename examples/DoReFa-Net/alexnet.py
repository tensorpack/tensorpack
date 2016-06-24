#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# Author: Yuheng Zou, Yuxin Wu {zouyuheng,wyx}@megvii.com

import cv2
import tensorflow as tf
import argparse
import numpy as np
import os

from tensorpack import *
from tensorpack.utils.stat import RatioCounter
from tensorpack.tfutils.symbolic_functions import prediction_incorrect

class Model(ModelDesc):
    def _get_input_vars(self):
        return [InputVar(tf.float32, [None, 224, 224, 3], 'input'),
                InputVar(tf.int32, [None], 'label') ]

    def _build_graph(self, input_vars, _):
        x, label = input_vars
        x = x / 256.0

        def quantize(x, name=None):
            # quantize to 2 bit
            return ((x * 3.0 + 0.5) // 1) / 3.0

        bn = lambda x, name: BatchNorm('bn', x, False, epsilon=1e-4)
        bnc = lambda x, name: tf.clip_by_value(bn(x, None), 0.0, 1.0, name=name)

        def conv_split(name, x, channel, shape):
            inputs = tf.split(3, 2, x)
            x0 = Conv2D(name + 'a', inputs[0], channel/2, shape)
            x1 = Conv2D(name + 'b', inputs[1], channel/2, shape)
            return tf.concat(3, [x0, x1])

        with argscope([Conv2D, FullyConnected], nl=bnc):
            x = Conv2D('conv1_1', x, 96, 12, stride=4, padding='VALID')
            x = quantize(x)
            x = conv_split('conv2_1', x, 256, 5)
            x = tf.pad(x, [[0,0], [1,1], [1,1], [0,0]])
            x = MaxPooling('pool1', x, 3, 2)
            x = quantize(x)

            x = Conv2D('conv3_1', x, 384, 3)
            x = tf.pad(x, [[0,0], [1,1], [1,1], [0,0]])
            x = MaxPooling('pool2', x, 3, 2)
            x = quantize(x)

            x = conv_split('conv4_1', x, 384, 3)
            x = quantize(x)

            x = conv_split('conv5_1', x, 256, 3)
            x = MaxPooling('pool3', x, 3, 2)
            x = quantize(x)
            x = tf.transpose(x, perm=[0,3,1,2])

            x = tf.nn.dropout(x, keep_prob=1.)
            x = FullyConnected('fc0', x, out_dim=4096)
            x = quantize(x)
            x = tf.nn.dropout(x, keep_prob=1.)
            x = FullyConnected('fc1', x, out_dim=4096)
            logits = FullyConnected('fct', x, out_dim=1000, nl=bn)

        prob = tf.nn.softmax(logits, name='prob')
        nr_wrong = tf.reduce_sum(prediction_incorrect(logits, label), name='wrong-top1')
        nr_wrong = tf.reduce_sum(prediction_incorrect(logits, label, 5), name='wrong-top5')

def eval_on_ILSVRC12(model, sess_init, data_dir):
    ds = dataset.ILSVRC12(data_dir, 'val', shuffle=False)

    def resize_func(im):
        h, w = im.shape[:2]
        scale = 256.0 / min(h, w)
        desSize = map(int, (max(224, min(w, scale * w)),\
                            max(224, min(h, scale * h))))
        im = cv2.resize(im, tuple(desSize), interpolation=cv2.INTER_CUBIC)
        return im
    transformers = [
        imgaug.MapImage(resize_func),
        imgaug.CenterCrop((224, 224)),
    ]
    ds = AugmentImageComponent(ds, transformers)
    ds = BatchData(ds, 128, remainder=True)
    ds = PrefetchData(ds, 10, 1)

    cfg = PredictConfig(
        model=model,
        session_init=sess_init,
        session_config=get_default_sess_config(0.99),
        output_var_names=['prob:0', 'wrong-top1:0', 'wrong-top5:0']
    )
    pred = SimpleDatasetPredictor(cfg, ds)

    acc1, acc5 = RatioCounter(), RatioCounter()
    for idx, o in enumerate(pred.get_result()):
        output, w1, w5 = o
        batch_size = output.shape[0]
        acc1.feed(w1, batch_size)
        acc5.feed(w5, batch_size)
    print("Top1 Error: {}".format(acc1.ratio))
    print("Top5 Error: {}".format(acc5.ratio))

def run_test(model, sess_init, inputs):
    pred_config = PredictConfig(
        model=model,
        input_data_mapping=[0],
        session_init=sess_init,
        session_config=get_default_sess_config(0.9),
        output_var_names=['prob:0']
    )
    predict_func = get_predict_func(pred_config)
    for f in inputs:
        assert os.path.isfile(f)
        img = cv2.imread(f)
        assert img is not None
        img = cv2.resize(img, (224, 224))[np.newaxis,:,:,:]
        outputs = predict_func([img])[0]
        prob = outputs[0]
        ret = prob.argsort()[-10:][::-1]

        meta = dataset.ILSVRCMeta().get_synset_words_1000()
        names = [meta[i] for i in ret]
        print(f + ":")
        print(list(zip(names, prob[ret])))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', help='path to the saved model parameters', required=True)
    parser.add_argument('--graph',
            help='path to the saved TF MetaGraph proto file. Used together with the model in TF format')
    parser.add_argument('--data', help='ILSVRC data directory. It must contains a subdirectory named \'val\'')
    parser.add_argument('--input', nargs='*', help='input images')
    args = parser.parse_args()

    if args.graph:
        # load graph definition
        M = ModelFromMetaGraph(args.graph)
    else:
        # build the graph from scratch
        logger.warn("[DoReFa-Net] Building the graph from scratch might result \
in compatibility issues in the future, if TensorFlow changes some of its \
op/variable names")
        M = Model()

    if args.load.endswith('.npy'):
        # load from a parameter dict
        param_dict= np.load(args.load, encoding='latin1').item()
        sess_init = ParamRestore(param_dict)
    elif args.load.endswith('.tfmodel'):
        sess_init = SaverRestore(args.load)
    else:
        raise RuntimeError("Unsupported model type!")

    if args.data:
        assert os.path.isdir(os.path.join(args.data, 'val'))
        eval_on_ILSVRC12(M, sess_init, args.data)
    elif args.input:
        run_test(M, sess_init, args.input)
    else:
        logger.error("Use '--data' to eval on ILSVRC, or '--input' to classify images")

