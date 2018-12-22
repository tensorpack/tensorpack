#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: mnist-embeddings.py

import argparse
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

from tensorpack import *
from tensorpack.tfutils.summary import add_moving_summary
from tensorpack.utils.gpu import change_gpu

from embedding_data import MnistPairs, MnistTriplets, get_test_data

MATPLOTLIB_AVAIBLABLE = False
try:
    from matplotlib import offsetbox
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAIBLABLE = True
except ImportError:
    MATPLOTLIB_AVAIBLABLE = False


def contrastive_loss(left, right, y, margin, extra=False, scope="constrastive_loss"):
    r"""Loss for Siamese networks as described in the paper:
    `Learning a Similarity Metric Discriminatively, with Application to Face
    Verification <http://yann.lecun.com/exdb/publis/pdf/chopra-05.pdf>`_ by Chopra et al.

    .. math::
        \frac{1}{2} [y \cdot d^2 + (1-y) \cdot \max(0, m - d)^2], d = \Vert l - r \Vert_2

    Args:
        left (tf.Tensor): left feature vectors of shape [Batch, N].
        right (tf.Tensor): right feature vectors of shape [Batch, N].
        y (tf.Tensor): binary labels of shape [Batch]. 1: similar, 0: not similar.
        margin (float): horizon for negative examples (y==0).
        extra (bool): also return distances for pos and neg.

    Returns:
        tf.Tensor: constrastive_loss (averaged over the batch), (and optionally average_pos_dist, average_neg_dist)
    """
    with tf.name_scope(scope):
        y = tf.cast(y, tf.float32)

        delta = tf.reduce_sum(tf.square(left - right), 1)
        delta_sqrt = tf.sqrt(delta + 1e-10)

        match_loss = delta
        missmatch_loss = tf.square(tf.nn.relu(margin - delta_sqrt))

        loss = tf.reduce_mean(0.5 * (y * match_loss + (1 - y) * missmatch_loss))

        if extra:
            num_pos = tf.count_nonzero(y)
            num_neg = tf.count_nonzero(1 - y)
            pos_dist = tf.where(tf.equal(num_pos, 0), 0.,
                                tf.reduce_sum(y * delta_sqrt) / tf.cast(num_pos, tf.float32),
                                name="pos-dist")
            neg_dist = tf.where(tf.equal(num_neg, 0), 0.,
                                tf.reduce_sum((1 - y) * delta_sqrt) / tf.cast(num_neg, tf.float32),
                                name="neg-dist")
            return loss, pos_dist, neg_dist
        else:
            return loss


def siamese_cosine_loss(left, right, y, scope="cosine_loss"):
    r"""Loss for Siamese networks (cosine version).
    Same as :func:`contrastive_loss` but with different similarity measurement.

    .. math::
        [\frac{l \cdot r}{\lVert l\rVert \lVert r\rVert} - (2y-1)]^2

    Args:
        left (tf.Tensor): left feature vectors of shape [Batch, N].
        right (tf.Tensor): right feature vectors of shape [Batch, N].
        y (tf.Tensor): binary labels of shape [Batch]. 1: similar, 0: not similar.

    Returns:
        tf.Tensor: cosine-loss as a scalar tensor.
    """

    def l2_norm(t, eps=1e-12):
        """
        Returns:
            tf.Tensor: norm of 2D input tensor on axis 1
        """
        with tf.name_scope("l2_norm"):
            return tf.sqrt(tf.reduce_sum(tf.square(t), 1) + eps)

    with tf.name_scope(scope):
        y = 2 * tf.cast(y, tf.float32) - 1
        pred = tf.reduce_sum(left * right, 1) / (l2_norm(left) * l2_norm(right) + 1e-10)

        return tf.nn.l2_loss(y - pred) / tf.cast(tf.shape(left)[0], tf.float32)


def triplet_loss(anchor, positive, negative, margin, extra=False, scope="triplet_loss"):
    r"""Loss for Triplet networks as described in the paper:
    `FaceNet: A Unified Embedding for Face Recognition and Clustering
    <https://arxiv.org/abs/1503.03832>`_
    by Schroff et al.

    Learn embeddings from an anchor point and a similar input (positive) as
    well as a not-similar input (negative).
    Intuitively, a matching pair (anchor, positive) should have a smaller relative distance
    than a non-matching pair (anchor, negative).

    .. math::
        \max(0, m + \Vert a-p\Vert^2 - \Vert a-n\Vert^2)

    Args:
        anchor (tf.Tensor): anchor feature vectors of shape [Batch, N].
        positive (tf.Tensor): features of positive match of the same shape.
        negative (tf.Tensor): features of negative match of the same shape.
        margin (float): horizon for negative examples
        extra (bool): also return distances for pos and neg.

    Returns:
        tf.Tensor: triplet-loss as scalar (and optionally average_pos_dist, average_neg_dist)
    """

    with tf.name_scope(scope):
        d_pos = tf.reduce_sum(tf.square(anchor - positive), 1)
        d_neg = tf.reduce_sum(tf.square(anchor - negative), 1)

        loss = tf.reduce_mean(tf.maximum(0., margin + d_pos - d_neg))

        if extra:
            pos_dist = tf.reduce_mean(tf.sqrt(d_pos + 1e-10), name='pos-dist')
            neg_dist = tf.reduce_mean(tf.sqrt(d_neg + 1e-10), name='neg-dist')
            return loss, pos_dist, neg_dist
        else:
            return loss


def soft_triplet_loss(anchor, positive, negative, extra=True, scope="soft_triplet_loss"):
    r"""Loss for triplet networks as described in the paper:
    `Deep Metric Learning using Triplet Network
    <https://arxiv.org/abs/1412.6622>`_ by Hoffer et al.

    It is a softmax loss using :math:`(anchor-positive)^2` and
    :math:`(anchor-negative)^2` as logits.

    Args:
        anchor (tf.Tensor): anchor feature vectors of shape [Batch, N].
        positive (tf.Tensor): features of positive match of the same shape.
        negative (tf.Tensor): features of negative match of the same shape.
        extra (bool): also return distances for pos and neg.

    Returns:
        tf.Tensor: triplet-loss as scalar (and optionally average_pos_dist, average_neg_dist)
    """

    eps = 1e-10
    with tf.name_scope(scope):
        d_pos = tf.sqrt(tf.reduce_sum(tf.square(anchor - positive), 1) + eps)
        d_neg = tf.sqrt(tf.reduce_sum(tf.square(anchor - negative), 1) + eps)

        logits = tf.stack([d_pos, d_neg], axis=1)
        ones = tf.ones_like(tf.squeeze(d_pos), dtype="int32")

        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=ones))

        if extra:
            pos_dist = tf.reduce_mean(d_pos, name='pos-dist')
            neg_dist = tf.reduce_mean(d_neg, name='neg-dist')
            return loss, pos_dist, neg_dist
        else:
            return loss


def center_loss(embedding, label, num_classes, alpha=0.1, scope="center_loss"):
    r"""Center-Loss as described in the paper
    `A Discriminative Feature Learning Approach for Deep Face Recognition`
    <http://ydwen.github.io/papers/WenECCV16.pdf> by Wen et al.

    Args:
        embedding (tf.Tensor): features produced by the network
        label (tf.Tensor): ground-truth label for each feature
        num_classes (int): number of different classes
        alpha (float): learning rate for updating the centers

    Returns:
        tf.Tensor: center loss
    """
    nrof_features = embedding.get_shape()[1]
    centers = tf.get_variable('centers', [num_classes, nrof_features], dtype=tf.float32,
                              initializer=tf.constant_initializer(0), trainable=False)
    label = tf.reshape(label, [-1])
    centers_batch = tf.gather(centers, label)
    diff = (1 - alpha) * (centers_batch - embedding)
    centers = tf.scatter_sub(centers, label, diff)
    loss = tf.reduce_mean(tf.square(embedding - centers_batch), name=scope)
    return loss


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

    def optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=1e-4, trainable=False)
        return tf.train.GradientDescentOptimizer(lr)


class SiameseModel(EmbeddingModel):
    @staticmethod
    def get_data():
        ds = MnistPairs('train')
        ds = BatchData(ds, 128 // 2)
        return ds

    def inputs(self):
        return [tf.placeholder(tf.float32, (None, 28, 28), 'input'),
                tf.placeholder(tf.float32, (None, 28, 28), 'input_y'),
                tf.placeholder(tf.int32, (None,), 'label')]

    def build_graph(self, x, y, label):
        # embed them
        single_input = x
        x, y = self.embed([x, y])

        # tag the embedding of 'input' with name 'emb', just for inference later on
        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
            tf.identity(self.embed(single_input), name="emb")

        # compute the actual loss
        cost, pos_dist, neg_dist = contrastive_loss(x, y, label, 5., extra=True, scope="loss")
        cost = tf.identity(cost, name="cost")

        # track these values during training
        add_moving_summary(pos_dist, neg_dist, cost)
        return cost


class CosineModel(SiameseModel):
    def build_graph(self, x, y, label):
        single_input = x
        x, y = self.embed([x, y])

        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
            tf.identity(self.embed(single_input), name="emb")

        cost = siamese_cosine_loss(x, y, label, scope="loss")
        cost = tf.identity(cost, name="cost")
        add_moving_summary(cost)
        return cost


class TripletModel(EmbeddingModel):
    @staticmethod
    def get_data():
        ds = MnistTriplets('train')
        ds = BatchData(ds, 128 // 3)
        return ds

    def inputs(self):
        return [tf.placeholder(tf.float32, (None, 28, 28), 'input'),
                tf.placeholder(tf.float32, (None, 28, 28), 'input_p'),
                tf.placeholder(tf.float32, (None, 28, 28), 'input_n')]

    def loss(self, a, p, n):
        return triplet_loss(a, p, n, 5., extra=True, scope="loss")

    def build_graph(self, a, p, n):
        single_input = a
        a, p, n = self.embed([a, p, n])

        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
            tf.identity(self.embed(single_input), name="emb")

        cost, pos_dist, neg_dist = self.loss(a, p, n)

        cost = tf.identity(cost, name="cost")
        add_moving_summary(pos_dist, neg_dist, cost)
        return cost


class SoftTripletModel(TripletModel):
    def loss(self, a, p, n):
        return soft_triplet_loss(a, p, n, scope="loss")


class CenterModel(EmbeddingModel):
    @staticmethod
    def get_data():
        ds = dataset.Mnist('train')
        ds = BatchData(ds, 128)
        return ds

    def inputs(self):
        return [tf.placeholder(tf.float32, (None, 28, 28), 'input'),
                tf.placeholder(tf.int32, (None,), 'label')]

    def build_graph(self, x, label):
        # embed them
        x = self.embed(x)
        x = tf.identity(x, name='emb')

        # compute the embedding loss
        emb_cost = center_loss(x, label, 10, 0.01)
        # compute the classification loss
        logits = slim.layers.fully_connected(tf.nn.relu(x), 10, activation_fn=None, scope='logits')

        cls_cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label),
                                  name='classification_costs')
        total_cost = tf.add(emb_cost, 100 * cls_cost, name="cost")

        # track these values during training
        add_moving_summary(total_cost, cls_cost, emb_cost)
        return total_cost


def get_config(model, algorithm_name):

    extra_display = ["cost"]
    if not algorithm_name == "cosine" and not algorithm_name == "center":
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
            RunUpdateOps()
        ],
        max_epoch=20,
    )


def visualize(model_path, model, algo_name):
    if not MATPLOTLIB_AVAIBLABLE:
        logger.error("visualize requires matplotlib package ...")
        return
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

    for offset, dp in enumerate(ds):
        digit, label = dp
        prediction = pred(digit)[0]
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
    plt.title('Embedding using %s-loss' % algo_name)
    plt.savefig('%s.jpg' % algo_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--load', help='load model')
    parser.add_argument('-a', '--algorithm', help='used algorithm', required=True,
                        choices=["siamese", "cosine", "triplet", "softtriplet", "center"])
    parser.add_argument('--visualize', help='export embeddings into an image', action='store_true')
    args = parser.parse_args()

    ALGO_CONFIGS = {"siamese": SiameseModel,
                    "cosine": CosineModel,
                    "triplet": TripletModel,
                    "softtriplet": SoftTripletModel,
                    "center": CenterModel}

    logger.auto_set_dir(name=args.algorithm)

    with change_gpu(args.gpu):
        if args.visualize:
            visualize(args.load, ALGO_CONFIGS[args.algorithm], args.algorithm)
        else:
            config = get_config(ALGO_CONFIGS[args.algorithm], args.algorithm)
            if args.load:
                config.session_init = SaverRestore(args.load)
            else:
                launch_train_with_config(config, SimpleTrainer())
