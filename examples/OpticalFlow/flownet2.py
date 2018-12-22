#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Patrick Wieschollek <mail@patwie.com>

import argparse
import glob
import os
import cv2

from tensorpack import *
from tensorpack.utils import viz

import flownet_models as models
from helper import Flow


def apply(model, model_path, left, right, ground_truth=None):
    left = cv2.imread(left)
    right = cv2.imread(right)

    h, w = left.shape[:2]
    newh = (h // 64) * 64
    neww = (w // 64) * 64
    aug = imgaug.CenterCrop((newh, neww))
    left, right = aug.augment(left), aug.augment(right)

    predict_func = OfflinePredictor(PredictConfig(
        model=model(height=newh, width=neww),
        session_init=get_model_loader(model_path),
        input_names=['left', 'right'],
        output_names=['prediction']))

    left_input, right_input = [x.astype('float32').transpose(2, 0, 1)[None, ...]
                               for x in [left, right]]
    output = predict_func(left_input, right_input)[0].transpose(0, 2, 3, 1)
    flow = Flow()

    img = flow.visualize(output[0])
    patches = [left, right, img * 255.]
    if ground_truth is not None:
        patches.append(flow.visualize(Flow.read(ground_truth)) * 255.)
    img = viz.stack_patches(patches, 2, 2)

    cv2.imshow('flow output', img)
    cv2.imwrite('flow_prediction.png', img)
    cv2.waitKey(0)


class SintelData(DataFlow):

    def __init__(self, data_path):
        super(SintelData, self).__init__()
        self.data_path = data_path
        self.path_prefix = os.path.join(data_path, 'flow')
        assert os.path.isdir(self.path_prefix), self.path_prefix
        self.flows = glob.glob(os.path.join(self.path_prefix, '*', '*.flo'))

    def size(self):
        return len(self.flows)

    def __iter__(self):
        for flow_path in self.flows:
            input_path = flow_path.replace(
                self.path_prefix, os.path.join(self.data_path, 'clean'))
            frame_id = int(input_path[-8:-4])
            input_a_path = '%s%04i.png' % (input_path[:-8], frame_id)
            input_b_path = '%s%04i.png' % (input_path[:-8], frame_id + 1)

            input_a = cv2.imread(input_a_path)
            input_b = cv2.imread(input_b_path)
            flow = Flow.read(flow_path)

            # most implementation just crop the center
            # which seems to be accepted practise
            h, w = input_a.shape[:2]
            newh = (h // 64) * 64
            neww = (w // 64) * 64
            aug = imgaug.CenterCrop((newh, neww))
            input_a = aug.augment(input_a)
            input_b = aug.augment(input_b)
            flow = aug.augment(flow)
            yield [input_a, input_b, flow]


def inference(model, model_path, sintel_path):
    ds = SintelData(sintel_path)

    def nhwc2nchw(dp):
        return [dp[0].transpose(2, 0, 1),
                dp[1].transpose(2, 0, 1),
                dp[2].transpose(2, 0, 1)]

    ds = MapData(ds, nhwc2nchw)
    ds = BatchData(ds, 1)
    ds.reset_state()

    # look at shape information (all images in Sintel has the same shape)
    h, w = next(ds.__iter__())[0].shape[2:]

    pred = PredictConfig(
        model=model(height=h, width=w),
        session_init=get_model_loader(model_path),
        input_names=['left', 'right', 'gt_flow'],
        output_names=['epe', 'prediction'])
    pred = SimpleDatasetPredictor(pred, ds)

    avg_epe, count_epe = 0, 0

    for o in pred.get_result():
        avg_epe += o[0]
        count_epe += 1

    print('average endpoint error (AEE): %f' % (float(avg_epe) / float(count_epe)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', help='path to the model', required=True)
    parser.add_argument('--model', help='model',
                        choices=['flownet2', 'flownet2-s', 'flownet2-c'], required=True)
    parser.add_argument('--left', help='input')
    parser.add_argument('--right', help='input')
    parser.add_argument('--gt', help='path to ground truth flow')
    parser.add_argument('--sintel_path', help='path to sintel dataset')
    args = parser.parse_args()

    model = {'flownet2-s': models.FlowNet2S,
             'flownet2-c': models.FlowNet2C,
             'flownet2': models.FlowNet2}[args.model]

    if args.sintel_path:
        inference(model, args.load, args.sintel_path)
    else:
        apply(model, args.load, args.left, args.right, args.gt)
