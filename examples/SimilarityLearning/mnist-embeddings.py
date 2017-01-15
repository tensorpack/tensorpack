#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: mnist-addition.py
# Author: PatWie <mail@patwie.com>

import tensorflow as tf
from tensorflow.python.platform import flags
import tensorflow.contrib.slim as slim

import numpy as np

import matplotlib
matplotlib.use("AGG")
import matplotlib.pyplot as plt  # noqa
from matplotlib import offsetbox  # noqa

from tensorpack import * # noqa
import tensorpack.tfutils.symbolic_functions as symbf # noqa
from tensorpack.tfutils.summary import add_moving_summary # noqa

FLAGS = flags.FLAGS
tf.app.flags.DEFINE_string('load', "", 'load model')
tf.app.flags.DEFINE_integer('gpu', 0, 'used gpu')
tf.app.flags.DEFINE_string('algorithm', "siamese", 'algorithm')
tf.app.flags.DEFINE_boolean('visualize', False, 'show embedding')
tf.app.flags.DEFINE_float('lr', 1e-4, 'show embedding')


class EmbeddingModel(ModelDesc):
    @staticmethod
    def get_data():
        ds = dataset.Mnist('test')
        ds = MapData(ds, lambda dp: dp)
        ds = BatchData(ds, 128)
        return ds

    def embed(self, x, nfeatures=2):
        """Embed all given tensors into an nfeatures-dim space.
        """
        list_split = 0
        if isinstance(x, list):
            list_split = len(x)
            x = tf.concat_v2(x, 0)

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
        # note the call has changed in v0.12 "split(value, num_or_size_splits, axis=0, ...)"
        if list_split > 0:
            embeddings = tf.split(embeddings, list_split, 0)

        return embeddings

    def _get_input_vars(self):
        return [InputVar(tf.float32, (None, 28, 28), 'input')]

    def _build_graph(self, input_vars):
        tf.identity(self.embed(input_vars), name="emb")


class MnistPairs(dataset.Mnist):
    """We could also write
        ds = dataset.Mnist('train')
        ds = JoinData([ds, ds])
        ds = MapData(ds, lambda dp: [dp[0], dp[2], dp[1] == dp[3]])
        ds = BatchData(ds, 128 // 2)

    but then the positives pairs would be really rare (p=0.1).
    Note, we do not override the method size() for practical considerations.
    """
    def __init__(self, train_or_test, shuffle=True, dir=None):
        super(MnistPairs, self).__init__(train_or_test, shuffle, dir)
        # now categorize these digits
        self.data_dict = []
        for clazz in range(0, 10):
            clazz_filter = np.where(self.labels == clazz)
            self.data_dict.append(self.images[clazz_filter])

    def get_data(self):
        while True:
            pick_label = self.rng.randint(10)
            pick_other = pick_label
            y = self.rng.randint(2)

            if y == 0:
                # pair with different digits
                offset = self.rng.randint(9)
                pick_other = (pick_label + offset + 1) % 10
                assert not pick_label == pick_other

            l = self.rng.randint(len(self.data_dict[pick_label]))
            r = self.rng.randint(len(self.data_dict[pick_other]))

            l = np.reshape(self.data_dict[pick_label][l], [28, 28]).astype(np.float32)
            r = np.reshape(self.data_dict[pick_other][r], [28, 28]).astype(np.float32)

            yield [l, r, y]


class SiameseModel(EmbeddingModel):
    @staticmethod
    def get_data():
        # return [digit_A, digit_B, similar?]
        # return [digit_A, digit_B, similar?]
        ds = MnistPairs('train')
        ds = BatchData(ds, 128 // 2)
        return ds

    def _get_input_vars(self):
        return [InputVar(tf.float32, (None, 28, 28), 'input'),
                InputVar(tf.float32, (None, 28, 28), 'input_y'),
                InputVar(tf.int32, (None,), 'label')]

    def _build_graph(self, input_vars):
        # get inputs
        x, y, label = input_vars
        # embed them
        x, y = self.embed([x, y])
        # tag them, just for inference later on
        x = tf.identity(x, name="emb")

        # compute the actual loss
        cost, pos_dist, neg_dist = symbf.contrastive_loss(x, y, label, 5., extra=True)
        self.cost = tf.identity(cost, name="cost")

        # track these values during training
        add_moving_summary(pos_dist)
        add_moving_summary(neg_dist)


class CosineModel(SiameseModel):

    def _build_graph(self, input_vars):
        # get inputs
        x, y, label = input_vars
        # embed them
        x, y = self.embed([x, y])
        # tag them, just for inference later on
        x = tf.identity(x, name="emb")

        # compute the actual loss
        cost = symbf.cosine_loss(x, y, label)
        self.cost = tf.identity(cost, name="cost")


class MnistTriplets(dataset.Mnist):
    """Deriving from dataset.Mnist is easier, than do it this processing on-the-fly
    """
    def __init__(self, train_or_test, shuffle=True, dir=None):
        super(MnistTriplets, self).__init__(train_or_test, shuffle, dir)

        # now categorize these digits
        self.data_dict = []
        for clazz in range(0, 10):
            clazz_filter = np.where(self.labels == clazz)
            self.data_dict.append(self.images[clazz_filter])

    def get_data(self):
        while True:
            pick_label = self.rng.randint(10)
            offset = self.rng.randint(9)
            pick_other = (pick_label + offset + 1) % 10
            assert not pick_label == pick_other

            a = self.rng.randint(len(self.data_dict[pick_label]))
            p = self.rng.randint(len(self.data_dict[pick_label]))
            n = self.rng.randint(len(self.data_dict[pick_other]))

            a = np.reshape(self.data_dict[pick_label][a], [28, 28]).astype(np.float32)
            p = np.reshape(self.data_dict[pick_label][p], [28, 28]).astype(np.float32)
            n = np.reshape(self.data_dict[pick_other][n], [28, 28]).astype(np.float32)

            yield [a, p, n]


class TripletModel(EmbeddingModel):
    @staticmethod
    def get_data():
        # return [digit_A, digit_B, similar?]
        ds = MnistTriplets('train')
        ds = BatchData(ds, 128 // 3)
        return ds

    def _get_input_vars(self):
        return [InputVar(tf.float32, (None, 28, 28), 'input'),
                InputVar(tf.float32, (None, 28, 28), 'input_p'),
                InputVar(tf.float32, (None, 28, 28), 'input_n')]

    def loss(self, a, p, n):
        return symbf.triplet_loss(a, p, n, 5., extra=True)

    def _build_graph(self, input_vars):
        # get inputs
        a, p, n = input_vars
        # embed them
        a, p, n = self.embed([a, p, n])
        # tag them, just for inference later on
        a = tf.identity(a, name="emb")

        # compute the actual loss
        cost, pos_dist, neg_dist = self.loss(a, p, n)
        self.cost = tf.identity(cost, name="cost")

        # track these values during training
        tf.summary.scalar('pos_dist', pos_dist)
        tf.summary.scalar('neg_dist', neg_dist)


class SoftTripletModel(TripletModel):
    def loss(self, a, p, n):
        return symbf.soft_triplet_loss(a, p, n)


def get_config(model):
    logger.auto_set_dir()

    dataset = model.get_data()
    step_per_epoch = dataset.size()

    lr = symbolic_functions.get_scalar_var('learning_rate', 1e-4, summary=True)

    return TrainConfig(
        dataflow=dataset,
        optimizer=tf.train.GradientDescentOptimizer(lr),
        callbacks=Callbacks([
            StatPrinter(),
            ModelSaver(),
            ScheduledHyperParamSetter('learning_rate', [(10, 1e-5), (20, 1e-6)])
        ]),
        model=model(),
        step_per_epoch=step_per_epoch,
        max_epoch=20,
    )


def visualize(modelpath, img_name="output"):
    pred = OfflinePredictor(PredictConfig(
        session_init=get_model_loader(modelpath),
        model=EmbeddingModel(),
        input_names=['input'],
        output_names=['emb']))

    NUM_BATCHES = 6
    BATCH_SIZE = 128
    images = np.zeros((BATCH_SIZE * NUM_BATCHES, 28, 28))  # the used digits
    embed = np.zeros((BATCH_SIZE * NUM_BATCHES, 2))  # the actual embeddings in 2-d

    # get only the embedding model data (MNIST test)
    ds = EmbeddingModel.get_data()
    ds.reset_state()

    offset = 0
    for digit, label in ds.get_data():
        # TODO: investigate why [0][0]
        prediction = pred([digit])[0][0]
        embed[offset * BATCH_SIZE:offset * BATCH_SIZE + BATCH_SIZE, ...] = prediction
        images[offset * BATCH_SIZE:offset * BATCH_SIZE + BATCH_SIZE, ...] = digit
        NUM_BATCHES -= 1
        offset += 1
        if NUM_BATCHES == 0:
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
    plt.title('Embedding using %s-loss' % img_name)
    plt.savefig('%s.jpg' % img_name)


if __name__ == '__main__':

    FLAGS = flags.FLAGS   # noqa
    FLAGS._parse_flags()

    assert FLAGS.algorithm in ["siamese", "cosine", "triplet", "softtriplet"]

    algorithm_configs = dict({"siamese": SiameseModel,
                              "cosine": CosineModel,
                              "triplet": TripletModel,
                              "softtriplet": SoftTripletModel})

    with change_gpu(FLAGS.gpu):
        config = get_config(algorithm_configs[FLAGS.algorithm])

        if FLAGS.load:
            config.session_init = SaverRestore(FLAGS.load)
        if FLAGS.visualize:
            visualize(FLAGS.load, FLAGS.algorithm)
        else:
            SimpleTrainer(config).train()
