#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import sys
import os
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.nets import resnet_v1

import tensorpack as tp
import tensorpack.utils.viz as viz

IMAGE_SIZE = 224


class Model(tp.ModelDesc):
    def _get_inputs(self):
        return [tp.InputDesc(tf.float32, (IMAGE_SIZE, IMAGE_SIZE, 3), 'image')]

    def _build_graph(self, inputs):
        orig_image = inputs[0]
        mean = tf.get_variable('resnet_v1_50/mean_rgb', shape=[3])
        with tp.symbolic_functions.guided_relu():
            with slim.arg_scope(resnet_v1.resnet_arg_scope(is_training=False)):
                image = tf.expand_dims(orig_image - mean, 0)
                logits, _ = resnet_v1.resnet_v1_50(image, 1000)
            tp.symbolic_functions.saliency_map(logits, orig_image, name="saliency")


def run(model_path, image_path):
    predictor = tp.OfflinePredictor(tp.PredictConfig(
        model=Model(),
        session_init=tp.get_model_loader(model_path),
        input_names=['image'],
        output_names=['saliency']))
    im = cv2.imread(image_path)
    assert im is not None and im.ndim == 3, image_path

    # resnet expect RGB inputs of 224x224x3
    im = cv2.resize(im, (IMAGE_SIZE, IMAGE_SIZE))
    im = im.astype(np.float32)[:, :, ::-1]

    saliency_images = predictor([im])[0]

    abs_saliency = np.abs(saliency_images).max(axis=-1)
    pos_saliency = np.maximum(0, saliency_images)
    neg_saliency = np.maximum(0, -saliency_images)

    pos_saliency -= pos_saliency.min()
    pos_saliency /= pos_saliency.max()
    cv2.imwrite('pos.jpg', pos_saliency * 255)

    neg_saliency -= neg_saliency.min()
    neg_saliency /= neg_saliency.max()
    cv2.imwrite('neg.jpg', neg_saliency * 255)

    abs_saliency = viz.intensity_to_rgb(abs_saliency, normalize=True)[:, :, ::-1]  # bgr
    cv2.imwrite("abs-saliency.jpg", abs_saliency)

    rsl = im * 0.2 + abs_saliency * 0.8
    cv2.imwrite("blended.jpg", rsl)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        tp.logger.error("Usage: {} image.jpg".format(sys.argv[0]))
        sys.exit(1)
    run("resnet_v1_50.ckpt", sys.argv[1])
