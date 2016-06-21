#!/usr/bin/env python2
# -*- coding: UTF-8 -*-
# Author: Yuheng Zou, Yuxin Wu {zouyuheng,wyx}@megvii.com

import cv2
import tensorflow as tf
import argparse
import numpy as np
import os

"""
Run the pretrained model of paper:
DoReFa-Net: Training Low Bitwidth Convolutional Neural Networks with Low Bitwidth Gradients
http://arxiv.org/abs/1606.06160

Model can be downloaded at:
https://drive.google.com/drive/u/2/folders/0B308TeQzmFDLa0xOeVQwcXg1ZjQ
"""

from tensorpack import *
from tensorpack.utils.stat import RatioCounter
from tensorpack.tfutils.symbolic_functions import prediction_incorrect

class Model(ModelDesc):
    def _get_input_vars(self):
        return [InputVar(tf.float32, [None, 224, 224, 3], 'input'),
                InputVar(tf.int32, [None], 'label') ]

    def _build_graph(self, input_vars, _):
        x, label = input_vars
        x = x / 255.0

        def tanh_round_bit(x, name=None):
            x = tf.tanh(x) * 0.5
            return (((x + 0.5) * 3.0 + 0.5) // 1) / 3.0 - 0.5

        x = Conv2D('conv1_1', x, 96, 12, nl=tanh_round_bit, stride=4, padding='VALID')

        bnl = lambda x, name: BatchNorm('bn', x, False, epsilon=1e-4)
        with argscope([Conv2D, FullyConnected], nl=bnl):
            x = Conv2D('conv2_1', x, 256, 5, padding='SAME')
            x = tf.pad(x, [[0,0], [1,1], [1,1], [0,0]], "SYMMETRIC")
            x = MaxPooling('pool1', x, 3, stride=2, padding='VALID')
            x = tanh_round_bit(x)

            x = Conv2D('conv3_1', x, 384, 3)
            x = tf.pad(x, [[0,0], [1,1], [1,1], [0,0]], "SYMMETRIC")
            x = MaxPooling('pool2', x, 3, stride=2, padding='VALID')
            x = tanh_round_bit(x)

            x = Conv2D('conv4_1', x, 384, 3)
            x = tanh_round_bit(x)

            x = Conv2D('conv5_1', x, 256, 3)
            x = MaxPooling('pool3', x, 3, stride=2, padding='VALID')
            x = tanh_round_bit(x)
            x = tf.transpose(x, perm=[0,3,1,2])

            x = FullyConnected('fc0', x, out_dim=4096)
            x = tanh_round_bit(x)
            x = FullyConnected('fc1', x, out_dim=4096)
            x = tf.tanh(x) * 0.5
            logits = FullyConnected('fct', x, out_dim=1000)

        prob = tf.nn.softmax(logits, name='prob')
        nr_wrong = tf.reduce_sum(prediction_incorrect(logits, label), name='wrong-top1')
        nr_wrong = tf.reduce_sum(prediction_incorrect(logits, label, 5), name='wrong-top5')



def eval_on_ILSVRC12(model, sess_init, data_dir):
    ds = dataset.ILSVRC12(data_dir, 'val', shuffle=False)
    transformers = [
        imgaug.Resize((256, 256)),
        imgaug.CenterCrop((224, 224)),
    ]
    ds = AugmentImageComponent(ds, transformers)
    ds = BatchData(ds, 128, remainder=True)
    ds = PrefetchDataZMQ(ds, 10)    # TODO use PrefetchData as fallback

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
        if idx == 10:
            print("Top1 Error: {} after {} images".format(acc1.ratio, acc1.count))
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
        print f + ":"
        print list(zip(names, prob[ret]))

    # save the metagraph
    #saver = tf.train.Saver()
    #saver.export_meta_graph('graph.meta', collection_list=
        #[INPUT_VARS_KEY, tf.GraphKeys.VARIABLES, tf.GraphKeys.TRAINABLE_VARIABLES], as_text=True)
    #saver.save(predict_func.session, 'alexnet.tfmodel', write_meta_graph=False)

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
        logger.warn("Building the graph from scratch might result \
in compatibility issues in the future, if TensorFlow changes some of its \
op/variable names")
        M = Model()

    if args.load.endswith('.npy'):
        # load from a parameter dict
        param_dict= np.load(args.load).item()
        sess_init = ParamRestore(param_dict)
    elif args.load.endswith('.tfmodel'):
        sess_init = SaverRestore(args.load)
    else:
        raise RuntimeError("Unsupported model type!")

    if args.data:
        eval_on_ILSVRC12(M, sess_init, args.data)
    elif args.input:
        run_test(M, sess_init, args.input)
    else:
        logger.error("Use '--data' to eval on ILSVRC, or '--input' to classify images")

