#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: model.py

import numpy as np
import tensorflow as tf
from tensorpack.tfutils import get_current_tower_context
from tensorpack.tfutils.summary import add_moving_summary
from tensorpack.tfutils.argscope import argscope
from tensorpack.tfutils.scope_utils import under_name_scope
from tensorpack.models import Conv2D, FullyConnected

from utils.box_ops import pairwise_iou
import config


def rpn_head(featuremap):
    with tf.variable_scope('rpn'), \
            argscope(Conv2D, data_format='NCHW',
                     W_init=tf.random_normal_initializer(stddev=0.01)):
        hidden = Conv2D('conv0', featuremap, 1024, 3, nl=tf.nn.relu)

        label_logits = Conv2D('class', hidden, config.NR_ANCHOR, 1)
        box_logits = Conv2D('box', hidden, 4 * config.NR_ANCHOR, 1)
        # 1, NA(*4), im/16, im/16 (NCHW)

        label_logits = tf.transpose(label_logits, [0, 2, 3, 1])  # 1xfHxfWxNA
        label_logits = tf.squeeze(label_logits, 0)  # fHxfWxNA

        shp = tf.shape(box_logits)  # 1x(NAx4)xfHxfW
        box_logits = tf.transpose(box_logits, [0, 2, 3, 1])  # 1xfHxfWx(NAx4)
        box_logits = tf.reshape(box_logits, tf.stack([shp[2], shp[3], config.NR_ANCHOR, 4]))  # fHxfWxNAx4
    return label_logits, box_logits


@under_name_scope()
def rpn_losses(anchor_labels, anchor_boxes, label_logits, box_logits):
    """
    Args:
        anchor_labels: fHxfWxNA
        anchor_boxes: fHxfWxNAx4, encoded
        label_logits:  fHxfWxNA
        box_logits: fHxfWxNAx4

    Returns:
        label_loss, box_loss
    """
    with tf.device('/cpu:0'):
        valid_mask = tf.stop_gradient(tf.not_equal(anchor_labels, -1))
        pos_mask = tf.stop_gradient(tf.equal(anchor_labels, 1))
        nr_valid = tf.stop_gradient(tf.count_nonzero(valid_mask), name='num_valid_anchor')
        nr_pos = tf.count_nonzero(pos_mask, name='num_pos_anchor')

        valid_anchor_labels = tf.boolean_mask(anchor_labels, valid_mask)
    valid_label_logits = tf.boolean_mask(label_logits, valid_mask)

    with tf.name_scope('label_metrics'):
        valid_label_prob = tf.nn.sigmoid(valid_label_logits)
        summaries = []
        with tf.device('/cpu:0'):
            for th in [0.5, 0.2, 0.1]:
                valid_prediction = tf.cast(valid_label_prob > th, tf.int32)
                prediction_corr = tf.count_nonzero(tf.equal(valid_prediction, valid_anchor_labels))
                pos_prediction_corr = tf.count_nonzero(tf.logical_and(
                    valid_label_prob > th,
                    tf.equal(valid_prediction, valid_anchor_labels)))
                summaries.append(tf.truediv(
                    pos_prediction_corr,
                    nr_pos, name='recall_th{}'.format(th)))
                summaries.append(tf.truediv(
                    prediction_corr,
                    nr_valid, name='accuracy_th{}'.format(th)))

    label_loss = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.to_float(valid_anchor_labels), logits=valid_label_logits)
    label_loss = tf.reduce_mean(label_loss, name='label_loss')

    pos_anchor_boxes = tf.boolean_mask(anchor_boxes, pos_mask)
    pos_box_logits = tf.boolean_mask(box_logits, pos_mask)
    delta = 1.0 / 9
    box_loss = tf.losses.huber_loss(
        pos_anchor_boxes, pos_box_logits, delta=delta,
        reduction=tf.losses.Reduction.SUM) / delta
    box_loss = tf.div(
        box_loss,
        tf.cast(nr_valid, tf.float32), name='box_loss')

    for k in [label_loss, box_loss, nr_valid, nr_pos] + summaries:
        add_moving_summary(k)
    return label_loss, box_loss


@under_name_scope()
def decode_bbox_target(box_predictions, anchors):
    """
    Args:
        box_predictions: fHxfWxNAx4, logits
        anchors: fHxfWxNAx4, floatbox

    Returns:
        box_decoded: (fHxfWxNA)x4, float32
    """
    box_pred_txtytwth = tf.reshape(box_predictions, (-1, 2, 2))
    box_pred_txty, box_pred_twth = tf.split(box_pred_txtytwth, 2, axis=1)
    # each is (fHxfWxNA)x1x2
    anchors_x1y1x2y2 = tf.reshape(anchors, (-1, 2, 2))
    anchors_x1y1, anchors_x2y2 = tf.split(anchors_x1y1x2y2, 2, axis=1)

    waha = tf.to_float(anchors_x2y2 - anchors_x1y1)
    xaya = tf.to_float(anchors_x2y2 + anchors_x1y1) * 0.5

    wbhb = tf.exp(tf.minimum(
        box_pred_twth, np.log(config.MAX_SIZE * 1.0 / config.ANCHOR_STRIDE))) * waha
    xbyb = box_pred_txty * waha + xaya
    x1y1 = xbyb - wbhb * 0.5
    x2y2 = xbyb + wbhb * 0.5
    out = tf.squeeze(tf.concat([x1y1, x2y2], axis=2), axis=1, name='output')
    return out


@under_name_scope()
def encode_bbox_target(boxes, anchors):
    """
    Args:
        boxes: fHxfWxNAx4, float32
        anchors: fHxfWxNAx4, float32

    Returns:
        box_encoded: fHxfWxNAx4
    """
    anchors_x1y1x2y2 = tf.reshape(anchors, (-1, 2, 2))
    anchors_x1y1, anchors_x2y2 = tf.split(anchors_x1y1x2y2, 2, axis=1)
    waha = tf.to_float(anchors_x2y2 - anchors_x1y1)
    xaya = tf.to_float(anchors_x2y2 + anchors_x1y1) * 0.5

    boxes_x1y1x2y2 = tf.reshape(boxes, (-1, 2, 2))
    boxes_x1y1, boxes_x2y2 = tf.split(boxes_x1y1x2y2, 2, axis=1)
    wbhb = tf.to_float(boxes_x2y2 - boxes_x1y1)
    xbyb = tf.to_float(boxes_x2y2 + boxes_x1y1) * 0.5

    # Note that here not all boxes are valid. Some may be zero

    txty = (xbyb - xaya) / waha
    twth = tf.log(wbhb / waha)  # may contain -inf for invalid boxes
    encoded = tf.concat([txty, twth], axis=1)  # (-1x2x2)
    return tf.reshape(encoded, tf.shape(boxes))


@under_name_scope()
def generate_rpn_proposals(boxes, scores, img_shape):
    """
    Args:
        boxes: nx4 float dtype, decoded to floatbox already
        scores: n float, the logits
        img_shape: [h, w]

    Returns:
        boxes: kx4 float
        scores: k logits
    """
    if get_current_tower_context().is_training:
        PRE_NMS_TOPK = config.TRAIN_PRE_NMS_TOPK
        POST_NMS_TOPK = config.TRAIN_POST_NMS_TOPK
    else:
        PRE_NMS_TOPK = config.TEST_PRE_NMS_TOPK
        POST_NMS_TOPK = config.TEST_POST_NMS_TOPK

    @under_name_scope()
    def clip_boxes(boxes, window):
        boxes = tf.maximum(boxes, 0.0)
        m = tf.tile(tf.reverse(window, [0]), [2])    # (4,)
        boxes = tf.minimum(boxes, tf.to_float(m))
        return boxes

    topk = tf.minimum(PRE_NMS_TOPK, tf.size(scores))
    topk_scores, topk_indices = tf.nn.top_k(scores, k=topk, sorted=False)
    topk_boxes = tf.gather(boxes, topk_indices)
    topk_boxes = clip_boxes(topk_boxes, img_shape)

    topk_boxes_x1y1x2y2 = tf.reshape(topk_boxes, (-1, 2, 2))
    topk_boxes_x1y1, topk_boxes_x2y2 = tf.split(topk_boxes_x1y1x2y2, 2, axis=1)
    # nx1x2 each
    wbhb = tf.squeeze(topk_boxes_x2y2 - topk_boxes_x1y1, axis=1)
    valid = tf.reduce_all(wbhb > config.RPN_MIN_SIZE, axis=1)  # n,
    topk_valid_boxes_x1y1x2y2 = tf.boolean_mask(topk_boxes_x1y1x2y2, valid)
    topk_valid_scores = tf.boolean_mask(topk_scores, valid)

    topk_valid_boxes_y1x1y2x2 = tf.reshape(
        tf.reverse(topk_valid_boxes_x1y1x2y2, axis=[2]),
        (-1, 4), name='nms_input_boxes')
    nms_indices = tf.image.non_max_suppression(
        topk_valid_boxes_y1x1y2x2,
        topk_valid_scores,
        max_output_size=POST_NMS_TOPK,
        iou_threshold=config.RPN_PROPOSAL_NMS_THRESH)

    topk_valid_boxes = tf.reshape(topk_valid_boxes_x1y1x2y2, (-1, 4))
    final_boxes = tf.gather(
        topk_valid_boxes,
        nms_indices, name='boxes')
    final_scores = tf.gather(topk_valid_scores, nms_indices, name='scores')
    final_probs = tf.gather(topk_valid_scores, nms_indices, name='probs')
    return final_boxes, final_scores


@under_name_scope()
def sample_fast_rcnn_targets(boxes, gt_boxes, gt_labels):
    """
    Args:
        boxes: nx4 region proposals, floatbox
        gt_boxes: mx4, floatbox
        gt_labels: m, int32

    Returns:
        sampled_boxes: tx4 floatbox, the rois
        target_boxes: tx4 encoded box, the regression target
        labels: t labels
    """
    @under_name_scope()
    def assign_class_to_roi(iou, gt_boxes, gt_labels):
        """
        Args:
            iou: nxm (nr_proposal x nr_gt)
        Returns:
            fg_mask: n boolean, whether each roibox is fg
            roi_labels: n int32, best label for each roi box
            best_gt_boxes: nx4
        """
        # find best gt box for each roi box
        best_iou_ind = tf.argmax(iou, axis=1)   # n, each in 1~m
        best_iou = tf.reduce_max(iou, axis=1)   # n,
        best_gt_boxes = tf.gather(gt_boxes, best_iou_ind)   # nx4
        best_gt_labels = tf.gather(gt_labels, best_iou_ind)     # n, each in 1~C

        fg_mask = best_iou >= config.FASTRCNN_FG_THRESH
        return fg_mask, best_gt_labels, best_gt_boxes

    iou = pairwise_iou(boxes, gt_boxes)     # nxm

    with tf.name_scope('proposal_metrics'):
        # find best roi for each gt, for summary only
        best_iou = tf.reduce_max(iou, axis=0)
        mean_best_iou = tf.reduce_mean(best_iou, name='best_iou_per_gt')
        summaries = [mean_best_iou]
        with tf.device('/cpu:0'):
            for th in [0.3, 0.5]:
                recall = tf.truediv(
                    tf.count_nonzero(best_iou >= th),
                    tf.size(best_iou, out_type=tf.int64),
                    name='recall_iou{}'.format(th))
                summaries.append(recall)
        add_moving_summary(*summaries)

    # n, n, nx4
    fg_mask, roi_labels, best_gt_boxes = assign_class_to_roi(iou, gt_boxes, gt_labels)

    # don't have to add gt for training, but add it anyway
    fg_inds = tf.reshape(tf.where(fg_mask), [-1])
    fg_inds = tf.concat([fg_inds, tf.cast(
        tf.range(tf.size(gt_labels)) + tf.shape(boxes)[0],
        tf.int64)], 0)
    num_fg = tf.size(fg_inds)
    num_fg = tf.minimum(int(
        config.FASTRCNN_BATCH_PER_IM * config.FASTRCNN_FG_RATIO),
        num_fg, name='num_fg')
    fg_inds = tf.slice(tf.random_shuffle(fg_inds), [0], [num_fg])

    bg_inds = tf.where(tf.logical_not(fg_mask))[:, 0]
    num_bg = tf.size(bg_inds)
    num_bg = tf.minimum(config.FASTRCNN_BATCH_PER_IM - num_fg, num_bg, name='num_bg')
    bg_inds = tf.slice(tf.random_shuffle(bg_inds), [0], [num_bg])

    add_moving_summary(num_fg, num_bg)

    all_boxes = tf.concat([boxes, gt_boxes], axis=0)
    all_matched_gt_boxes = tf.concat([best_gt_boxes, gt_boxes], axis=0)
    all_labels = tf.concat([roi_labels, gt_labels], axis=0)

    ind_in_all = tf.concat([fg_inds, bg_inds], axis=0)   # ind in all n+m boxes
    ret_boxes = tf.gather(all_boxes, ind_in_all, name='sampled_boxes')
    ret_matched_gt_boxes = tf.gather(all_matched_gt_boxes, ind_in_all)
    ret_encoded_boxes = encode_bbox_target(ret_matched_gt_boxes, ret_boxes)
    ret_encoded_boxes = ret_encoded_boxes * tf.constant(config.FASTRCNN_BBOX_REG_WEIGHTS)
    # bg boxes will not be trained on

    ret_labels = tf.concat(
        [tf.gather(all_labels, fg_inds),
         tf.zeros_like(bg_inds, dtype=tf.int64)], axis=0, name='sampled_labels')
    return ret_boxes, tf.stop_gradient(ret_encoded_boxes), tf.stop_gradient(ret_labels)


@under_name_scope()
def roi_align(featuremap, boxes, output_shape):
    """
    Args:
        featuremap: 1xCxHxW
        boxes: Nx4 floatbox
        output_shape: int

    Returns:
        NxCxoHxoW
    """
    @under_name_scope()
    def transform_fpcoor_for_tf(boxes, image_shape, crop_shape):
        """
        The way crop_and_resize works (with normalized box):
        Initial point (the value of output[0]): x0_box * (W_img - 1)
        Spacing: w_box * (W_img - 1) / (W_crop - 1)
        Use the above grid to bilinear sample.

        However, what I want is (with fpcoor box):
        Spacing: w_box / W_crop
        Initial point: x0_box + spacing/2 - 0.5
        (-0.5 because bilinear sample assumes floating point coordinate (0.0, 0.0) is the same as pixel value (0, 0))

        This function transform fpcoor boxes to a format to be used by tf.image.crop_and_resize

        Returns:
            y1x1y2x2
        """
        x0, y0, x1, y1 = tf.split(boxes, 4, axis=1)

        spacing_w = (x1 - x0) / tf.to_float(crop_shape[1])
        spacing_h = (y1 - y0) / tf.to_float(crop_shape[0])

        nx0 = (x0 + spacing_w / 2 - 0.5) / tf.to_float(image_shape[1] - 1)
        ny0 = (y0 + spacing_h / 2 - 0.5) / tf.to_float(image_shape[0] - 1)

        nw = spacing_w * tf.to_float(crop_shape[1] - 1) / tf.to_float(image_shape[1] - 1)
        nh = spacing_h * tf.to_float(crop_shape[0] - 1) / tf.to_float(image_shape[0] - 1)

        return tf.concat([ny0, nx0, ny0 + nh, nx0 + nw], axis=1)

    image_shape = tf.shape(featuremap)[2:]
    featuremap = tf.transpose(featuremap, [0, 2, 3, 1])  # to nhwc
    # sample 4 locations per roi bin
    boxes = transform_fpcoor_for_tf(boxes, image_shape, [output_shape * 2, output_shape * 2])
    boxes = tf.stop_gradient(boxes)  # TODO
    ret = tf.image.crop_and_resize(
        featuremap, boxes, tf.zeros([tf.shape(boxes)[0]], dtype=tf.int32),
        crop_size=[output_shape * 2, output_shape * 2])
    ret = tf.transpose(ret, [0, 3, 1, 2])
    ret = tf.nn.avg_pool(ret, [1, 1, 2, 2], [1, 1, 2, 2], padding='SAME', data_format='NCHW')
    return ret


def fastrcnn_head(feature, num_classes):
    """
    Args:
        feature (NxCx1x1):
        num_classes(int): num_category + 1

    Returns:
        cls_logits (Nxnum_class), reg_logits (Nx num_class-1 x 4)
    """
    with tf.variable_scope('fastrcnn'):
        classification = FullyConnected(
            'class', feature, num_classes,
            W_init=tf.random_normal_initializer(stddev=0.01))
        box_regression = FullyConnected(
            'box', feature, (num_classes - 1) * 4,
            W_init=tf.random_normal_initializer(stddev=0.001))
        box_regression = tf.reshape(box_regression, (-1, num_classes - 1, 4))
        return classification, box_regression


@under_name_scope()
def fastrcnn_predict_boxes(labels, box_logits):
    """
    Args:
        labels: n,
        box_logits: nx(C-1)x4

    Returns:
        fg_ind: fg, indices into n
        fg_box_logits: fgx4

    """
    fg_ind = tf.reshape(tf.where(labels > 0), [-1])  # nfg,
    fg_labels = tf.gather(labels, fg_ind)   # nfg,

    ind_2d = tf.stack([fg_ind, fg_labels - 1], axis=1)  # nfgx2
    # n x c-1 x 4 -> nfgx4
    fg_box_logits = tf.gather_nd(box_logits, tf.stop_gradient(ind_2d))
    return fg_ind, fg_box_logits


@under_name_scope()
def fastrcnn_losses(labels, boxes, label_logits, box_logits):
    """
    Args:
        labels: n,
        boxes: nx4, encoded
        label_logits: nxC
        box_logits: nx(C-1)x4
    """
    label_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=label_logits)
    label_loss = tf.reduce_mean(label_loss, name='label_loss')
    prediction = tf.argmax(label_logits, axis=1, name='label_prediction')
    correct = tf.to_float(tf.equal(prediction, labels))  # boolean/integer gather is unavailable on GPU
    accuracy = tf.reduce_mean(correct, name='accuracy')

    # n x c-1 x 4 -> nfg x 4
    fg_ind, fg_box_logits = fastrcnn_predict_boxes(labels, box_logits)
    fg_boxes = tf.gather(boxes, fg_ind)  # nfgx4

    fg_label_pred = tf.argmax(tf.gather(label_logits, fg_ind), axis=1)
    num_zero = tf.reduce_sum(tf.cast(tf.equal(fg_label_pred, 0), tf.int32), name='num_zero')
    false_negative = tf.truediv(num_zero, tf.size(fg_ind), name='false_negative')
    fg_correct = tf.gather(correct, fg_ind)
    fg_accuracy = tf.reduce_mean(fg_correct, name='fg_accuracy')

    box_loss = tf.losses.huber_loss(
        fg_boxes, fg_box_logits, reduction=tf.losses.Reduction.SUM)
    box_loss = tf.truediv(
        box_loss, tf.to_float(tf.shape(labels)[0]), name='box_loss')

    for k in [label_loss, box_loss, accuracy, fg_accuracy, false_negative]:
        add_moving_summary(k)
    return label_loss, box_loss
