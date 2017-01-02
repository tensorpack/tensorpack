#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import tensorflow as tf
import cPickle as pickle
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.nets import resnet_v1
import os
import numpy as np

"""
Extracting Saliency using guided Relu
"""

import tensorpack as tp

IMAGE_SIZE = 224


class Model(tp.ModelDesc):
    def _get_input_vars(self):
        """Define all the input variables (with type, shape, name) that'll be
        fed into the graph to produce a cost.  """
        return [tp.InputVar(tf.float32, (1, IMAGE_SIZE, IMAGE_SIZE, 3), 'input'),
                tp.InputVar(tf.int32, (1,), 'label')]

    def _build_graph(self, input_vars):
        """This function should build the model which takes the input variables
        and define self.cost at the end"""

        # input_vars contains a list of input variables defined above
        image, label = input_vars

        # get model
        with tp.symbolic_functions.GuidedRelu():
            with slim.arg_scope(resnet_v1.resnet_arg_scope(is_training=False)):
                logits, _ = resnet_v1.resnet_v1_50(image, 1000)
            tp.symbolic_functions.saliency(logits, image, name="saliency")


def run(model_path, image_path):
    pred_config = tp.PredictConfig(model=Model(),
                                   session_init=tp.get_model_loader(model_path),
                                   input_names=['input'],
                                   output_names=['saliency'])
    predict_func = tp.get_predict_func(pred_config)
    im = cv2.imread(image_path)
    assert im is not None

    # ResNet expect some specific properties of an image
    with open("/graphics/projects/data/neuralnetworks/imagenet/tf_resnet/mean_bgr.p", "r") as hnd:
        mean_bgr = pickle.load(hnd)

    # resnet expect inputs of 224x224x3
    im = cv2.resize(im, (224, 224))
    # use float32
    im = np.array(im).astype(np.float32)
    # bgr -> bgr [c,h,w]
    im = im.transpose(2, 0, 1)
    # subtract mean_bgr
    im -= mean_bgr
    # [c, w, h] --> [ w, h, c]
    im = im.transpose(1, 2, 0)
    # now back to rgb
    im = im[:, :, [2, 1, 0]]

    saliency_images = predict_func([[im.astype('float32')]])[0]

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
    """
    You need to download a pre-trained model from here: https://github.com/tensorflow/models/tree/master/slim
    This example uses ResNet-50:

    wget http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz
    tar -xzvf resnet_v1_50_2016_08_28.tar.gz
    """

    os.environ['CUDA_VISIBLE_DEVICES'] = "0"

    run("resnet_v1_50.ckpt", "cat.jpg")
