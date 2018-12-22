#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import sys
from contextlib import contextmanager
import cv2
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.nets import resnet_v1

import tensorpack as tp
import tensorpack.utils.viz as viz

IMAGE_SIZE = 224


@contextmanager
def guided_relu():
    """
    Returns:
        A context where the gradient of :meth:`tf.nn.relu` is replaced by
        guided back-propagation, as described in the paper:
        `Striving for Simplicity: The All Convolutional Net
        <https://arxiv.org/abs/1412.6806>`_
    """
    from tensorflow.python.ops import gen_nn_ops   # noqa

    @tf.RegisterGradient("GuidedReLU")
    def GuidedReluGrad(op, grad):
        return tf.where(0. < grad,
                        gen_nn_ops._relu_grad(grad, op.outputs[0]),
                        tf.zeros(grad.get_shape()))

    g = tf.get_default_graph()
    with g.gradient_override_map({'Relu': 'GuidedReLU'}):
        yield


def saliency_map(output, input, name="saliency_map"):
    """
    Produce a saliency map as described in the paper:
    `Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps
    <https://arxiv.org/abs/1312.6034>`_.
    The saliency map is the gradient of the max element in output w.r.t input.

    Returns:
        tf.Tensor: the saliency map. Has the same shape as input.
    """
    max_outp = tf.reduce_max(output, 1)
    saliency_op = tf.gradients(max_outp, input)[:][0]
    return tf.identity(saliency_op, name=name)


class Model(tp.ModelDescBase):
    def inputs(self):
        return [tf.placeholder(tf.float32, (IMAGE_SIZE, IMAGE_SIZE, 3), 'image')]

    def build_graph(self, orig_image):
        mean = tf.get_variable('resnet_v1_50/mean_rgb', shape=[3])
        with guided_relu():
            with slim.arg_scope(resnet_v1.resnet_arg_scope(is_training=False)):
                image = tf.expand_dims(orig_image - mean, 0)
                logits, _ = resnet_v1.resnet_v1_50(image, 1000)
            saliency_map(logits, orig_image, name="saliency")


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

    saliency_images = predictor(im)[0]

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
