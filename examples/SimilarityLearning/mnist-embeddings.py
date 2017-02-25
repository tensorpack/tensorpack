#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: mnist-embeddings.py
# Author: PatWie <mail@patwie.com>

import tensorflow as tf
from tensorflow.python.platform import flags
import tensorflow.contrib.slim as slim

import numpy as np

import matplotlib
from matplotlib import offsetbox
import matplotlib.pyplot as plt

from tensorpack import *
import tensorpack.tfutils.symbolic_functions as symbf
from tensorpack.tfutils.summary import add_moving_summary
from embedding_data import get_test_data, MnistPairs, MnistTriplets


FLAGS = flags.FLAGS
tf.app.flags.DEFINE_string('load', "", 'load model')
tf.app.flags.DEFINE_integer('gpu', 0, 'used gpu')
tf.app.flags.DEFINE_string('algorithm', "siamese", 'algorithm')
tf.app.flags.DEFINE_boolean('visualize', False, 'show embedding')


class EmbeddingModel(ModelDesc):
    def embed(self, x, nfeatures=2):
        """Embed all given tensors into an nfeatures-dim space.  """
        list_split = 0
        if isinstance(x, list):
            list_split = len(x)
            x = tf.concat(x, 0)

        # pre-process MNIST dataflow data
        x = tf.expand_dims(x, 3)
        x = x * 2 - 1

        # the embedding network
        net = slim.layers.conv2d(x, 20, 5, scope='conv1')
        net = slim.layers.max_pool2d(net, 2, scope='pool1')
        net = slim.layers.conv2d(net, 50, 5, scope='conv2')
        net = slim.layers.max_pool2d(net, 2, scope='pool2')
        net = slim.layers.flatten(net, scope='flatten3')
        net = slim.layers.fully_connected(net, 500, scope='fully_connected4')
        embeddings = slim.layers.fully_connected(net, nfeatures, activation_fn=None, scope='fully_connected5')

        # if "x" was a list of tensors, then split the embeddings
        if list_split > 0:
            embeddings = tf.split(embeddings, list_split, 0)

        return embeddings

    def _get_optimizer(self):
        lr = symbf.get_scalar_var('learning_rate', 1e-4, summary=True)
        return tf.train.GradientDescentOptimizer(lr)


class SiameseModel(EmbeddingModel):
    @staticmethod
    def get_data():
        ds = MnistPairs('train')
        ds = BatchData(ds, 128 // 2)
        return ds

    def _get_inputs(self):
        return [InputDesc(tf.float32, (None, 28, 28), 'input'),
                InputDesc(tf.float32, (None, 28, 28), 'input_y'),
                InputDesc(tf.int32, (None,), 'label')]

    def _build_graph(self, inputs):
        # get inputs
        x, y, label = inputs
        # embed them
        x, y = self.embed([x, y])

        # tag the embedding of 'input' with name 'emb', just for inference later on
        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
            tf.identity(self.embed(inputs[0]), name="emb")

        # compute the actual loss
        cost, pos_dist, neg_dist = symbf.contrastive_loss(x, y, label, 5., extra=True, scope="loss")
        self.cost = tf.identity(cost, name="cost")

        # track these values during training
        add_moving_summary(pos_dist, neg_dist, self.cost)


class CosineModel(SiameseModel):
    def _build_graph(self, inputs):
        x, y, label = inputs
        x, y = self.embed([x, y])

        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
            tf.identity(self.embed(inputs[0]), name="emb")

        cost = symbf.cosine_loss(x, y, label, scope="loss")
        self.cost = tf.identity(cost, name="cost")
        add_moving_summary(self.cost)


class TripletModel(EmbeddingModel):
    @staticmethod
    def get_data():
        ds = MnistTriplets('train')
        ds = BatchData(ds, 128 // 3)
        return ds

    def _get_inputs(self):
        return [InputDesc(tf.float32, (None, 28, 28), 'input'),
                InputDesc(tf.float32, (None, 28, 28), 'input_p'),
                InputDesc(tf.float32, (None, 28, 28), 'input_n')]

    def loss(self, a, p, n):
        return symbf.triplet_loss(a, p, n, 5., extra=True, scope="loss")

    def _build_graph(self, inputs):
        a, p, n = inputs
        a, p, n = self.embed([a, p, n])

        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
            tf.identity(self.embed(inputs[0]), name="emb")

        cost, pos_dist, neg_dist = self.loss(a, p, n)

        self.cost = tf.identity(cost, name="cost")
        add_moving_summary(pos_dist, neg_dist, self.cost)


class SoftTripletModel(TripletModel):
    def loss(self, a, p, n):
        return symbf.soft_triplet_loss(a, p, n, scope="loss")


def get_config(model, algorithm_name):
    logger.auto_set_dir()

    extra_display = ["cost"]
    if not algorithm_name == "cosine":
        extra_display = extra_display + ["loss/pos-dist", "loss/neg-dist"]

    return TrainConfig(
        dataflow=model.get_data(),
        model=model(),
        callbacks=[
            ModelSaver(),
            ScheduledHyperParamSetter('learning_rate', [(10, 1e-5), (20, 1e-6)])
        ],
        extra_callbacks=[
            MovingAverageSummary(),
            ProgressBar(extra_display),
            MergeAllSummaries(),
        ],
        max_epoch=20,
    )


def visualize(model_path, model):
    pred = OfflinePredictor(PredictConfig(
        session_init=get_model_loader(model_path),
        model=model(),
        input_names=['input'],
        output_names=['emb']))

    NUM_BATCHES = 6
    BATCH_SIZE = 128
    images = np.zeros((BATCH_SIZE * NUM_BATCHES, 28, 28))  # the used digits
    embed = np.zeros((BATCH_SIZE * NUM_BATCHES, 2))  # the actual embeddings in 2-d

    # get only the embedding model data (MNIST test)
    ds = get_test_data()
    ds.reset_state()

    for offset, dp in enumerate(ds.get_data()):
        digit, label = dp
        prediction = pred([digit])[0]
        embed[offset * BATCH_SIZE:offset * BATCH_SIZE + BATCH_SIZE, ...] = prediction
        images[offset * BATCH_SIZE:offset * BATCH_SIZE + BATCH_SIZE, ...] = digit
        offset += 1
        if offset == NUM_BATCHES:
            break

    plt.figure()
    ax = plt.subplot(111)
    ax_min = np.min(embed, 0)
    ax_max = np.max(embed, 0)

    ax_dist_sq = np.sum((ax_max - ax_min)**2)
    ax.axis('off')
    shown_images = np.array([[1., 1.]])
    for i in range(embed.shape[0]):
        dist = np.sum((embed[i] - shown_images)**2, 1)
        if np.min(dist) < 3e-4 * ax_dist_sq:     # don't show points that are too close
            continue
        shown_images = np.r_[shown_images, [embed[i]]]
        imagebox = offsetbox.AnnotationBbox(offsetbox.OffsetImage(np.reshape(images[i, ...], [28, 28]),
                                            zoom=0.6, cmap=plt.cm.gray_r), xy=embed[i], frameon=False)
        ax.add_artist(imagebox)

    plt.axis([ax_min[0], ax_max[0], ax_min[1], ax_max[1]])
    plt.xticks([]), plt.yticks([])
    algo_name = FLAGS.algorithm
    plt.title('Embedding using %s-loss' % algo_name)
    plt.savefig('%s.jpg' % algo_name)


if __name__ == '__main__':
    unknown = FLAGS._parse_flags()
    assert len(unknown) == 0, "Invalid argument!"
    assert FLAGS.algorithm in ["siamese", "cosine", "triplet", "softtriplet"]

    ALGO_CONFIGS = {"siamese": SiameseModel,
                    "cosine": CosineModel,
                    "triplet": TripletModel,
                    "softtriplet": SoftTripletModel}

    with change_gpu(FLAGS.gpu):
        if FLAGS.visualize:
            visualize(FLAGS.load, ALGO_CONFIGS[FLAGS.algorithm])
        else:
            config = get_config(ALGO_CONFIGS[FLAGS.algorithm], FLAGS.algorithm)
            if FLAGS.load:
                config.session_init = SaverRestore(FLAGS.load)
            else:
                SimpleTrainer(config).train()
