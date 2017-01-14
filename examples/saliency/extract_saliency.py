#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import cPickle as pickle
import sys
import os
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.nets import resnet_v1
import tensorpack as tp

"""
Extracting Saliency using guided ReLU
"""
IMAGE_SIZE = 224


def saliency(output, input, name="saliency"):
    """
    Returns:
        The gradient of the maximum number in output w.r.t input.
    """
    max_outp = tf.reduce_max(output, 1)
    saliency_op = tf.gradients(max_outp, input)[:][0]
    saliency_op = tf.identity(saliency_op, name=name)
    return saliency_op


class Model(tp.ModelDesc):
    def _get_input_vars(self):
        return [tp.InputVar(tf.float32, (1, IMAGE_SIZE, IMAGE_SIZE, 3), 'input'),
                tp.InputVar(tf.int32, (1,), 'label')]

    def _build_graph(self, input_vars):
        image, label = input_vars
        mean = tf.get_variable('resnet_v1_50/mean_rgb', shape=[3])
        image = image - mean
        with tp.symbolic_functions.GuidedReLU():
            with slim.arg_scope(resnet_v1.resnet_arg_scope(is_training=False)):
                logits, _ = resnet_v1.resnet_v1_50(image, 1000)
            saliency(logits, image, name="saliency")


def run(model_path, image_path):
    predict_func = tp.OfflinePredictor(tp.PredictConfig(
        model=Model(),
        session_init=tp.get_model_loader(model_path),
        input_names=['input'],
        output_names=['saliency']))
    im = cv2.imread(image_path)
    assert im is not None, image_path

    # resnet expect RGB inputs of 224x224x3
    im = cv2.resize(im, (IMAGE_SIZE, IMAGE_SIZE))
    im = im.astype(np.float32)[:, :, ::-1]

    saliency_images = predict_func([[im]])[0]

    abs_saliency = (1 - np.abs(saliency_images).max(axis=-1))
    abs_saliency -= abs_saliency.min()
    abs_saliency /= abs_saliency.max()
    pos_saliency = (np.maximum(0, saliency_images) / saliency_images.max())
    neg_saliency = (np.maximum(0, -saliency_images) / -saliency_images.min())

    abs_saliency = abs_saliency[0, :, :]
    pos_saliency = pos_saliency[0, :, :, [2, 1, 0]].transpose(1, 2, 0)
    neg_saliency = neg_saliency[0, :, :, [2, 1, 0]].transpose(1, 2, 0)

    cv2.imwrite("abs_saliency.jpg", tp.utils.intensity_to_rgb(abs_saliency, cmap='Blues') * 255.)
    cv2.imwrite("pos_saliency.jpg", tp.utils.intensity_to_rgb(pos_saliency, cmap='Blues') * 255.)
    cv2.imwrite("neg_saliency.jpg", tp.utils.intensity_to_rgb(neg_saliency, cmap='Blues') * 255.)

    intensity = 1 - abs_saliency

    highres_img = cv2.imread("cat.jpg")
    rsl = tp.utils.intensity_to_rgb(highres_img[:, :, 0], cmap='Blues')
    cv2.imwrite("intensity.jpg", rsl * 255)

    intensity = cv2.resize(intensity, (highres_img.shape[1], highres_img.shape[0]))
    rsl = tp.utils.filter_intensity(intensity, highres_img)
    cv2.imwrite("heatmap.jpg", rsl)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        tp.logger.error("Usage: {} image.jpg".format(sys.argv[0]))
        sys.exit(1)
    run("resnet_v1_50.ckpt", sys.argv[1])
