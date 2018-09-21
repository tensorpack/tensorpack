#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import cv2
import tensorflow as tf
from tensorpack import *
from tensorpack.tfutils.export import ModelExporter

"""
This example illustrates the process of exporting a model trained in Tensorpack to:
- npz containing just the weights
- TensorFlow Serving
- a frozen and pruned inference graph (compact)

The model applies a laplace filter to the input image.

The steps are:

1. train the model by

    python export.py --gpu 0

2. export the model by

    python export.py --export npz --load train_log/export/checkpoint
    python export.py --export serving --load train_log/export/checkpoint
    python export.py --export compact --load train_log/export/checkpoint

3. run inference by

    python export.py --apply default --load train_log/export/checkpoint
    python export.py --apply inference_graph --load train_log/export/checkpoint
    python export.py --apply compact --load /tmp/compact_graph.pb
"""


BATCH_SIZE = 1
SHAPE = 256
CHANNELS = 3


class Model(ModelDesc):
    """Just a simple model, which applies the Laplacian-operation to images to showcase
    the usage of variables, and alternating the inference-graph later.
    """

    def inputs(self):
        return [tf.placeholder(tf.uint8, (None, SHAPE, SHAPE, CHANNELS), 'input_img'),
                tf.placeholder(tf.uint8, (None, SHAPE, SHAPE, CHANNELS), 'target_img')]

    def make_prediction(self, img):

        img = tf.cast(img, tf.float32)
        img = tf.image.rgb_to_grayscale(img)

        k = tf.get_variable('filter', dtype=tf.float32,
                            initializer=[[[[0.]], [[1.]], [[0.]]], [
                                [[1.]], [[-4.]], [[1.]]], [[[0.]], [[1.]], [[0.]]]])
        prediction_img = tf.nn.conv2d(img, k, strides=[1, 1, 1, 1], padding='SAME')
        return prediction_img

    def build_graph(self, input_img, target_img):

        target_img = tf.cast(target_img, tf.float32)
        target_img = tf.image.rgb_to_grayscale(target_img)

        self.prediction_img = tf.identity(self.make_prediction(input_img), name='prediction_img')

        cost = tf.losses.mean_squared_error(target_img, self.prediction_img,
                                            reduction=tf.losses.Reduction.MEAN)
        return tf.identity(cost, name='total_costs')

    def optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=0.0, trainable=False)
        return tf.train.AdamOptimizer(lr)


def get_data(subset):
    """
    You might train this model using some fake data. But since the learning-rate is 0,
    it does not change the model weights.
    """
    ds = FakeData([[SHAPE, SHAPE, CHANNELS], [SHAPE, SHAPE, CHANNELS]], 1000, random=False,
                  dtype=['uint8', 'uint8'], domain=[(0, 255), (0, 10)])
    ds = BatchData(ds, BATCH_SIZE)
    return ds


def get_config():
    """
    The usual config for training.
    """
    logger.auto_set_dir()

    ds_train = get_data('train')

    return TrainConfig(
        model=Model(),
        data=QueueInput(ds_train),
        callbacks=[
            ModelSaver(),
        ],
        steps_per_epoch=1,
        max_epoch=1,
    )


class InferenceOnlyModel(Model):
    """This illustrates the way to rewrite the inference graph to accept images encoded as base64.

    Remarks:
        CloudML expects base64 encoded data to feed into placeholders with suffix "_bytes".
    """

    def inputs(self):
        # The inference graph only accepts a single image, which is different to the training model.
        return [tf.placeholder(tf.string, (None,), 'input_img_bytes')]

    def build_graph(self, input_img_bytes):
        # prepare input (base64 encoded strings to images)
        input_img = tf.map_fn(lambda x: tf.image.decode_png(x, channels=3), input_img_bytes, dtype=tf.uint8)

        # It is a good idea to just copy the inference relevant parts to this graph.
        prediction_img = self.make_prediction(input_img)

        # outputs should be base64 encoded strings agains
        prediction_img = tf.clip_by_value(prediction_img, 0, 255)
        prediction_img = tf.cast(prediction_img, tf.uint8)
        prediction_img_bytes = tf.map_fn(tf.image.encode_png, prediction_img, dtype=tf.string)

        # prediction_img_bytes = tf.image.encode_png(prediction_img)
        tf.identity(prediction_img_bytes, name='prediction_img_bytes')


def export_npz():
    """The default Tensorpack way of packing just the weights into numpy
       compressed containers.
    """
    print('RUN')
    print("python tensorpack/scripts/dump-model-params.py "
          "--meta train_log/export/graph-0904-161912.meta "
          "train_log/export/checkpoint "
          "weights.npz")


def export_serving(model_path):
    """Export trained model to use it in TensorFlow Serving or cloudML.
    """
    pred_config = PredictConfig(
        session_init=get_model_loader(model_path),
        model=InferenceOnlyModel(),
        input_names=['input_img_bytes'],
        output_names=['prediction_img_bytes'])
    ModelExporter(pred_config).export_serving('/tmp/exported')


def export_compact(model_path):
    """Export trained model to use it as a frozen and pruned inference graph in
       mobile applications.
    """
    pred_config = PredictConfig(
        session_init=get_model_loader(model_path),
        model=Model(),
        input_names=['input_img'],
        output_names=['prediction_img'])
    ModelExporter(pred_config).export_compact('/tmp/compact_graph.pb')


def apply(model_path):
    """Run inference from a training model checkpoint.
    """
    pred_config = PredictConfig(
        session_init=get_model_loader(model_path),
        model=Model(),
        input_names=['input_img'],
        output_names=['prediction_img'])

    pred = OfflinePredictor(pred_config)
    img = cv2.imread('lena.png')
    prediction = pred([img])[0]
    cv2.imwrite('applied_default.jpg', prediction[0])


def apply_inference_graph(model_path):
    """Run inference from an altered graph, which receives
       encoded images buffers.
    """
    pred_config = PredictConfig(
        session_init=get_model_loader(model_path),
        model=InferenceOnlyModel(),
        input_names=['input_img_bytes'],
        output_names=['prediction_img_bytes'])

    pred = OfflinePredictor(pred_config)

    with open('lena.png', 'rb') as f:
        buf = f.read()

    prediction = pred([buf])[0]

    with open('applied_inference_graph.png', 'wb') as f:
        f.write(prediction[0])


def apply_compact(graph_path):
    """Run pruned and frozen inference graph.
    """
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        # Note, we just load the graph and do *not* initialize anything.
        with tf.gfile.GFile(graph_path, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def)

        input_img = sess.graph.get_tensor_by_name('import/input_img:0')
        prediction_img = sess.graph.get_tensor_by_name('import/prediction_img:0')

        prediction = sess.run(prediction_img, {input_img: cv2.imread('lena.png')[None, ...]})
        cv2.imwrite('applied_compact.png', prediction[0])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--load', help='load model')
    parser.add_argument('--apply', help='run sampling', default='')
    parser.add_argument('--export', help='export the model', default='')

    args = parser.parse_args()

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if args.apply != '':
        assert args.apply in ['default', 'inference_graph', 'compact']
        if args.apply == 'default':
            apply(args.load)
        elif args.apply == 'inference_graph':
            apply_inference_graph(args.load)
        else:
            apply_compact(args.load)
    elif args.export != '':
        assert args.export in ['serving', 'compact', 'npz']
        if args.export == 'npz':
            export_npz()
        elif args.export == 'serving':
            export_serving(args.load)
        else:
            export_compact(args.load)
    else:
        config = get_config()

        if args.gpu:
            config.nr_tower = len(args.gpu.split(','))
        if args.load:
            config.session_init = SaverRestore(args.load)

        launch_train_with_config(config, SimpleTrainer())
