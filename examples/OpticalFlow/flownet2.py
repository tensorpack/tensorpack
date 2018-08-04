#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Patrick Wieschollek <mail@patwie.com>

import os
import cv2
import glob
from helper import Flow
import argparse
import tensorflow as tf
import numpy as np

from tensorpack import *

import flownet_models as models

enable_argscope_for_module(tf.layers)

"""
This is a tensorpack script re-implementation of
FlowNet 2.0: Evolution of Optical Flow Estimation with Deep Networks
https://arxiv.org/abs/1612.01925

This is not an attempt to reproduce the lengthly training protocol (fow now),
but to rely on tensorpack's "OfflinePredictor" for easier inference.

The ported pre-trained Caffe-model are here
http://models.tensorpack.com/opticalflow/flownet2-s.npz
http://models.tensorpack.com/opticalflow/flownet2-c.npz
http://models.tensorpack.com/opticalflow/flownet2.npz

It has the original license:

```
    Pre-trained weights are provided for research purposes only and without any warranty.

    Any commercial use of the pre-trained weights requires FlowNet2 authors consent.
    When using the the pre-trained weights in your research work, please cite the following paper:

    @InProceedings{IMKDB17,
      author       = "E. Ilg and N. Mayer and T. Saikia and M. Keuper and A. Dosovitskiy and T. Brox",
      title        = "FlowNet 2.0: Evolution of Optical Flow Estimation with Deep Networks",
      booktitle    = "IEEE Conference on Computer Vision and Pattern Recognition (CVPR)",
      month        = "Jul",
      year         = "2017",
      url          = "http://lmb.informatik.uni-freiburg.de//Publications/2017/IMKDB17"
    }
```

To run it on actual data:

    python flownet2.py --gpu 0 \
        --left 00001_img1.ppm \
        --right 00001_img2.ppm \
        --load flownet2.npz
        --model "flownet2"

To evaluate AEE on sintel:

    python flownet2.py --gpu 2 \
        --load flownet2.npz \
        --model "flownet2" \
        --sintel_path "/path/to/sintel/training/"

"""


MODEL_MAP = {'flownet2-s': models.FlowNet2S,
             'flownet2-c': models.FlowNet2C,
             'flownet2': models.FlowNet2}


def apply(model_name, model_path, left, right, ground_truth=None):
    model = MODEL_MAP[model_name]
    left = cv2.imread(left).astype(np.float32).transpose(2, 0, 1)[None, ...]
    right = cv2.imread(right).astype(np.float32).transpose(2, 0, 1)[None, ...]

    _, _, h, w = left.shape

    predict_func = OfflinePredictor(PredictConfig(
        model=model(height=h, width=w),
        session_init=get_model_loader(model_path),
        input_names=['left', 'right'],
        output_names=['prediction']))

    output = predict_func(left, right)[0].transpose(0, 2, 3, 1)
    flow = Flow()

    img = flow.visualize(output[0])
    if ground_truth is not None:
        img = np.concatenate([img, flow.visualize(Flow.read(ground_truth))], axis=1)

    cv2.imshow('flow output', img)
    cv2.imwrite('flow_prediction.png', img * 255)
    cv2.waitKey(0)


class SintelData(DataFlow):
    """Read images directly from tar file without unpacking.
    """

    def __init__(self, data_path):
        super(SintelData, self).__init__()
        assert os.path.isdir(data_path)
        self.data_path = data_path
        self.path_prefix = os.path.join(data_path, 'flow', )
        self.flows = glob.glob(os.path.join(self.path_prefix, '*', '*.flo'))

    def size(self):
        return len(self.flows)

    def get_data(self):
        for flow_path in self.flows:
            input_path = flow_path.replace(
                self.path_prefix, os.path.join(self.data_path, 'clean', ))
            frame_id = int(input_path[-8:-4])
            input_a_path = '%s%04i.png' % (input_path[:-8], frame_id)
            input_b_path = '%s%04i.png' % (input_path[:-8], frame_id + 1)

            input_a = cv2.imread(input_a_path)
            input_b = cv2.imread(input_b_path)
            flow = Flow.read(flow_path)

            # most implementation just crop the center
            # which seems to be accepted practise
            h, w = input_a.shape[:2]
            h_ = (h // 64) * 64
            w_ = (w // 64) * 64
            h_start = (h - h_) // 2
            w_start = (w - w_) // 2

            # this is ugly
            h_end = -h_start if h_start > 0 else h
            w_end = -w_start if w_start > 0 else w

            input_a = input_a[h_start:h_end, w_start:w_end, :]
            input_b = input_b[h_start:h_end, w_start:w_end, :]
            flow = flow[h_start:h_end, w_start:w_end, :]

            yield [input_a, input_b, flow]


def inference(model_name, model_path, sintel_path):
    assert os.path.isdir(sintel_path)
    model = MODEL_MAP[model_name]

    ds = SintelData(sintel_path)

    def nhwc2nchw(dp):
        return [dp[0].transpose(2, 0, 1),
                dp[1].transpose(2, 0, 1),
                dp[2].transpose(2, 0, 1)]

    ds = MapData(ds, nhwc2nchw)
    ds = BatchData(ds, 1)
    ds.reset_state()

    # look at shape information
    h, w = next(ds.get_data())[0].shape[2:]

    pred = PredictConfig(
        model=model(height=h, width=w),
        session_init=get_model_loader(model_path),
        input_names=['left', 'right', 'gt_flow'],
        output_names=['epe', 'prediction'])
    pred = SimpleDatasetPredictor(pred, ds)

    avg_epe = 0
    count_epe = 0

    for o in pred.get_result():
        avg_epe += o[0]
        count_epe += 1

    print('average endpoint error (AEE): %f' % (float(avg_epe) / float(count_epe)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--load', help='load model')
    parser.add_argument('--left', help='input', type=str)
    parser.add_argument('--right', help='input', type=str)
    parser.add_argument('--model', help='model', type=str)
    parser.add_argument('--sintel_path', help='path to sintel dataset', type=str)
    parser.add_argument('--gt', help='ground_truth', type=str, default=None)
    args = parser.parse_args()

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if args.sintel_path != '':
        inference(args.model, args.load, args.sintel_path)
    else:
        apply(args.model, args.load, args.left, args.right, args.gt)
