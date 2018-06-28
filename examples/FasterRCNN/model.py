# -*- coding: utf-8 -*-
# File: model.py

import tensorflow as tf
import numpy as np
import itertools

from tensorpack.tfutils.summary import add_moving_summary
from tensorpack.tfutils.argscope import argscope
from tensorpack.tfutils.scope_utils import under_name_scope, auto_reuse_variable_scope
from tensorpack.models import (
    Conv2D, FullyConnected, MaxPooling,
    layer_register, Conv2DTranspose, FixedUnPooling)

from utils.box_ops import pairwise_iou
from utils.box_ops import area as tf_area
from model_box import roi_align, clip_boxes
from config import config as cfg


@layer_register(log_shape=True)
@auto_reuse_variable_scope
def rpn_head(featuremap, channel, num_anchors):
    """
    Returns:
        label_logits: fHxfWxNA
        box_logits: fHxfWxNAx4
    """
    with argscope(Conv2D, data_format='channels_first',
                  kernel_initializer=tf.random_normal_initializer(stddev=0.01)):
        hidden = Conv2D('conv0', featuremap, channel, 3, activation=tf.nn.relu)

        label_logits = Conv2D('class', hidden, num_anchors, 1)
        box_logits = Conv2D('box', hidden, 4 * num_anchors, 1)
        # 1, NA(*4), im/16, im/16 (NCHW)

        label_logits = tf.transpose(label_logits, [0, 2, 3, 1])  # 1xfHxfWxNA
        label_logits = tf.squeeze(label_logits, 0)  # fHxfWxNA

        shp = tf.shape(box_logits)  # 1x(NAx4)xfHxfW
        box_logits = tf.transpose(box_logits, [0, 2, 3, 1])  # 1xfHxfWx(NAx4)
        box_logits = tf.reshape(box_logits, tf.stack([shp[2], shp[3], num_anchors, 4]))  # fHxfWxNAx4
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
        nr_valid = tf.stop_gradient(tf.count_nonzero(valid_mask, dtype=tf.int32), name='num_valid_anchor')
        nr_pos = tf.identity(tf.count_nonzero(pos_mask, dtype=tf.int32), name='num_pos_anchor')
        # nr_pos is guaranteed >0 in C4. But in FPN. even nr_valid could be 0.

        valid_anchor_labels = tf.boolean_mask(anchor_labels, valid_mask)
    valid_label_logits = tf.boolean_mask(label_logits, valid_mask)

    with tf.name_scope('label_metrics'):
        valid_label_prob = tf.nn.sigmoid(valid_label_logits)
        summaries = []
        with tf.device('/cpu:0'):
            for th in [0.5, 0.2, 0.1]:
                valid_prediction = tf.cast(valid_label_prob > th, tf.int32)
                nr_pos_prediction = tf.reduce_sum(valid_prediction, name='num_pos_prediction')
                pos_prediction_corr = tf.count_nonzero(
                    tf.logical_and(
                        valid_label_prob > th,
                        tf.equal(valid_prediction, valid_anchor_labels)),
                    dtype=tf.int32)
                placeholder = 0.5   # A small value will make summaries appear lower.
                recall = tf.to_float(tf.truediv(pos_prediction_corr, nr_pos))
                recall = tf.where(tf.equal(nr_pos, 0), placeholder, recall, name='recall_th{}'.format(th))
                precision = tf.to_float(tf.truediv(pos_prediction_corr, nr_pos_prediction))
                precision = tf.where(tf.equal(nr_pos_prediction, 0),
                                     placeholder, precision, name='precision_th{}'.format(th))
                summaries.extend([precision, recall])
        add_moving_summary(*summaries)

    # Per-level loss summaries in FPN may appear lower due to the use of a small placeholder.
    # But the total loss is still the same.  TODO make the summary op smarter
    placeholder = 0.
    label_loss = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=tf.to_float(valid_anchor_labels), logits=valid_label_logits)
    label_loss = tf.reduce_sum(label_loss) * (1. / cfg.RPN.BATCH_PER_IM)
    label_loss = tf.where(tf.equal(nr_valid, 0), placeholder, label_loss, name='label_loss')

    pos_anchor_boxes = tf.boolean_mask(anchor_boxes, pos_mask)
    pos_box_logits = tf.boolean_mask(box_logits, pos_mask)
    delta = 1.0 / 9
    box_loss = tf.losses.huber_loss(
        pos_anchor_boxes, pos_box_logits, delta=delta,
        reduction=tf.losses.Reduction.SUM) / delta
    box_loss = box_loss * (1. / cfg.RPN.BATCH_PER_IM)
    box_loss = tf.where(tf.equal(nr_pos, 0), placeholder, box_loss, name='box_loss')

    add_moving_summary(label_loss, box_loss, nr_valid, nr_pos)
    return label_loss, box_loss


@under_name_scope()
def generate_rpn_proposals(boxes, scores, img_shape,
                           pre_nms_topk, post_nms_topk=None):
    """
    Sample RPN proposals by the following steps:
    1. Pick top k1 by scores
    2. NMS them
    3. Pick top k2 by scores. Default k2 == k1, i.e. does not filter the NMS output.

    Args:
        boxes: nx4 float dtype, the proposal boxes. Decoded to floatbox already
        scores: n float, the logits
        img_shape: [h, w]
        pre_nms_topk, post_nms_topk (int): See above.

    Returns:
        boxes: kx4 float
        scores: k logits
    """
    assert boxes.shape.ndims == 2, boxes.shape
    if post_nms_topk is None:
        post_nms_topk = pre_nms_topk

    topk = tf.minimum(pre_nms_topk, tf.size(scores))
    topk_scores, topk_indices = tf.nn.top_k(scores, k=topk, sorted=False)
    topk_boxes = tf.gather(boxes, topk_indices)
    topk_boxes = clip_boxes(topk_boxes, img_shape)

    topk_boxes_x1y1x2y2 = tf.reshape(topk_boxes, (-1, 2, 2))
    topk_boxes_x1y1, topk_boxes_x2y2 = tf.split(topk_boxes_x1y1x2y2, 2, axis=1)
    # nx1x2 each
    wbhb = tf.squeeze(topk_boxes_x2y2 - topk_boxes_x1y1, axis=1)
    valid = tf.reduce_all(wbhb > cfg.RPN.MIN_SIZE, axis=1)  # n,
    topk_valid_boxes_x1y1x2y2 = tf.boolean_mask(topk_boxes_x1y1x2y2, valid)
    topk_valid_scores = tf.boolean_mask(topk_scores, valid)

    # TODO not needed
    topk_valid_boxes_y1x1y2x2 = tf.reshape(
        tf.reverse(topk_valid_boxes_x1y1x2y2, axis=[2]),
        (-1, 4), name='nms_input_boxes')
    nms_indices = tf.image.non_max_suppression(
        topk_valid_boxes_y1x1y2x2,
        # TODO use exp to work around a bug in TF1.9: https://github.com/tensorflow/tensorflow/issues/19578
        tf.exp(topk_valid_scores),
        max_output_size=post_nms_topk,
        iou_threshold=cfg.RPN.PROPOSAL_NMS_THRESH)

    topk_valid_boxes = tf.reshape(topk_valid_boxes_x1y1x2y2, (-1, 4))
    final_boxes = tf.gather(topk_valid_boxes, nms_indices)
    final_scores = tf.gather(topk_valid_scores, nms_indices)
    tf.sigmoid(final_scores, name='probs')  # for visualization
    return tf.stop_gradient(final_boxes, name='boxes'), tf.stop_gradient(final_scores, name='scores')


@under_name_scope()
def proposal_metrics(iou):
    """
    Add summaries for RPN proposals.

    Args:
        iou: nxm, #proposal x #gt
    """
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


@under_name_scope()
def sample_fast_rcnn_targets(boxes, gt_boxes, gt_labels):
    """
    Sample some ROIs from all proposals for training.
    #fg is guaranteed to be > 0, because grount truth boxes are added as RoIs.

    Args:
        boxes: nx4 region proposals, floatbox
        gt_boxes: mx4, floatbox
        gt_labels: m, int32

    Returns:
        sampled_boxes: tx4 floatbox, the rois
        sampled_labels: t labels, in [0, #class-1]. Positive means foreground.
        fg_inds_wrt_gt: #fg indices, each in range [0, m-1].
            It contains the matching GT of each foreground roi.
    """
    iou = pairwise_iou(boxes, gt_boxes)     # nxm
    proposal_metrics(iou)

    # add ground truth as proposals as well
    boxes = tf.concat([boxes, gt_boxes], axis=0)    # (n+m) x 4
    iou = tf.concat([iou, tf.eye(tf.shape(gt_boxes)[0])], axis=0)   # (n+m) x m
    # #proposal=n+m from now on

    def sample_fg_bg(iou):
        fg_mask = tf.reduce_max(iou, axis=1) >= cfg.FRCNN.FG_THRESH

        fg_inds = tf.reshape(tf.where(fg_mask), [-1])
        num_fg = tf.minimum(int(
            cfg.FRCNN.BATCH_PER_IM * cfg.FRCNN.FG_RATIO),
            tf.size(fg_inds), name='num_fg')
        fg_inds = tf.random_shuffle(fg_inds)[:num_fg]

        bg_inds = tf.reshape(tf.where(tf.logical_not(fg_mask)), [-1])
        num_bg = tf.minimum(
            cfg.FRCNN.BATCH_PER_IM - num_fg,
            tf.size(bg_inds), name='num_bg')
        bg_inds = tf.random_shuffle(bg_inds)[:num_bg]

        add_moving_summary(num_fg, num_bg)
        return fg_inds, bg_inds

    fg_inds, bg_inds = sample_fg_bg(iou)
    # fg,bg indices w.r.t proposals

    best_iou_ind = tf.argmax(iou, axis=1)   # #proposal, each in 0~m-1
    fg_inds_wrt_gt = tf.gather(best_iou_ind, fg_inds)   # num_fg

    all_indices = tf.concat([fg_inds, bg_inds], axis=0)   # indices w.r.t all n+m proposal boxes
    ret_boxes = tf.gather(boxes, all_indices)

    ret_labels = tf.concat(
        [tf.gather(gt_labels, fg_inds_wrt_gt),
         tf.zeros_like(bg_inds, dtype=tf.int64)], axis=0)
    # stop the gradient -- they are meant to be training targets
    return tf.stop_gradient(ret_boxes, name='sampled_proposal_boxes'), \
        tf.stop_gradient(ret_labels, name='sampled_labels'), \
        tf.stop_gradient(fg_inds_wrt_gt)


@layer_register(log_shape=True)
def fastrcnn_outputs(feature, num_classes):
    """
    Args:
        feature (any shape):
        num_classes(int): num_category + 1

    Returns:
        cls_logits (Nxnum_class), reg_logits (Nx num_class x 4)
    """
    classification = FullyConnected(
        'class', feature, num_classes,
        kernel_initializer=tf.random_normal_initializer(stddev=0.01))
    box_regression = FullyConnected(
        'box', feature, num_classes * 4,
        kernel_initializer=tf.random_normal_initializer(stddev=0.001))
    box_regression = tf.reshape(box_regression, (-1, num_classes, 4))
    return classification, box_regression


@layer_register(log_shape=True)
def fastrcnn_2fc_head(feature, num_classes):
    """
    Args:
        feature (any shape):
        num_classes(int): num_category + 1

    Returns:
        cls_logits (Nxnum_class), reg_logits (Nx num_class-1 x 4)
    """
    dim = cfg.FPN.FRCNN_FC_HEAD_DIM
    init = tf.variance_scaling_initializer()
    hidden = FullyConnected('fc6', feature, dim, kernel_initializer=init, activation=tf.nn.relu)
    hidden = FullyConnected('fc7', hidden, dim, kernel_initializer=init, activation=tf.nn.relu)
    return fastrcnn_outputs('outputs', hidden, num_classes)


@layer_register(log_shape=True)
def fastrcnn_Xconv1fc_head(feature, num_classes, num_convs):
    """
    Args:
        feature (any shape):
        num_classes(int): num_category + 1
        num_convs (int): number of conv layers

    Returns:
        cls_logits (Nxnum_class), reg_logits (Nx num_class-1 x 4)
    """
    l = feature
    with argscope(Conv2D, data_format='channels_first',
                  kernel_initializer=tf.variance_scaling_initializer(
                      scale=2.0, mode='fan_out', distribution='normal')):
        for k in range(num_convs):
            l = Conv2D('conv{}'.format(k), l, cfg.FPN.FRCNN_CONV_HEAD_DIM, 3, activation=tf.nn.relu)
        l = FullyConnected('fc', l, cfg.FPN.FRCNN_FC_HEAD_DIM,
                           kernel_initializer=tf.variance_scaling_initializer(), activation=tf.nn.relu)
    return fastrcnn_outputs('outputs', l, num_classes)


def fastrcnn_4conv1fc_head(*args, **kwargs):
    return fastrcnn_Xconv1fc_head(*args, num_convs=4, **kwargs)


@under_name_scope()
def fastrcnn_losses(labels, label_logits, fg_boxes, fg_box_logits):
    """
    Args:
        labels: n,
        label_logits: nxC
        fg_boxes: nfgx4, encoded
        fg_box_logits: nfgxCx4
    """
    label_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
        labels=labels, logits=label_logits)
    label_loss = tf.reduce_mean(label_loss, name='label_loss')

    fg_inds = tf.where(labels > 0)[:, 0]
    fg_labels = tf.gather(labels, fg_inds)
    num_fg = tf.size(fg_inds)
    indices = tf.stack(
        [tf.range(num_fg),
         tf.to_int32(fg_labels)], axis=1)  # #fgx2
    fg_box_logits = tf.gather_nd(fg_box_logits, indices)

    with tf.name_scope('label_metrics'), tf.device('/cpu:0'):
        prediction = tf.argmax(label_logits, axis=1, name='label_prediction')
        correct = tf.to_float(tf.equal(prediction, labels))  # boolean/integer gather is unavailable on GPU
        accuracy = tf.reduce_mean(correct, name='accuracy')
        fg_label_pred = tf.argmax(tf.gather(label_logits, fg_inds), axis=1)
        num_zero = tf.reduce_sum(tf.to_int32(tf.equal(fg_label_pred, 0)), name='num_zero')
        false_negative = tf.truediv(num_zero, num_fg, name='false_negative')
        fg_accuracy = tf.reduce_mean(
            tf.gather(correct, fg_inds), name='fg_accuracy')

    box_loss = tf.losses.huber_loss(
        fg_boxes, fg_box_logits, reduction=tf.losses.Reduction.SUM)
    box_loss = tf.truediv(
        box_loss, tf.to_float(tf.shape(labels)[0]), name='box_loss')

    add_moving_summary(label_loss, box_loss, accuracy, fg_accuracy, false_negative)
    return label_loss, box_loss


@under_name_scope()
def fastrcnn_predictions(boxes, probs):
    """
    Generate final results from predictions of all proposals.

    Args:
        boxes: n#catx4 floatbox in float32
        probs: nx#class
    """
    assert boxes.shape[1] == cfg.DATA.NUM_CATEGORY
    assert probs.shape[1] == cfg.DATA.NUM_CLASS
    boxes = tf.transpose(boxes, [1, 0, 2])  # #catxnx4
    probs = tf.transpose(probs[:, 1:], [1, 0])  # #catxn

    def f(X):
        """
        prob: n probabilities
        box: nx4 boxes

        Returns: n boolean, the selection
        """
        prob, box = X
        output_shape = tf.shape(prob)
        # filter by score threshold
        ids = tf.reshape(tf.where(prob > cfg.TEST.RESULT_SCORE_THRESH), [-1])
        prob = tf.gather(prob, ids)
        box = tf.gather(box, ids)
        # NMS within each class
        selection = tf.image.non_max_suppression(
            box, prob, cfg.TEST.RESULTS_PER_IM, cfg.TEST.FRCNN_NMS_THRESH)
        selection = tf.to_int32(tf.gather(ids, selection))
        # sort available in TF>1.4.0
        # sorted_selection = tf.contrib.framework.sort(selection, direction='ASCENDING')
        sorted_selection = -tf.nn.top_k(-selection, k=tf.size(selection))[0]
        mask = tf.sparse_to_dense(
            sparse_indices=sorted_selection,
            output_shape=output_shape,
            sparse_values=True,
            default_value=False)
        return mask

    masks = tf.map_fn(f, (probs, boxes), dtype=tf.bool,
                      parallel_iterations=10)     # #cat x N
    selected_indices = tf.where(masks)  # #selection x 2, each is (cat_id, box_id)
    probs = tf.boolean_mask(probs, masks)

    # filter again by sorting scores
    topk_probs, topk_indices = tf.nn.top_k(
        probs,
        tf.minimum(cfg.TEST.RESULTS_PER_IM, tf.size(probs)),
        sorted=False)
    filtered_selection = tf.gather(selected_indices, topk_indices)
    filtered_selection = tf.reverse(filtered_selection, axis=[1], name='filtered_indices')
    return filtered_selection, topk_probs


@layer_register(log_shape=True)
def maskrcnn_upXconv_head(feature, num_category, num_convs):
    """
    Args:
        feature (NxCx s x s): size is 7 in C4 models and 14 in FPN models.
        num_category(int):
        num_convs (int): number of convolution layers

    Returns:
        mask_logits (N x num_category x 2s x 2s):
    """
    l = feature
    with argscope([Conv2D, Conv2DTranspose], data_format='channels_first',
                  kernel_initializer=tf.variance_scaling_initializer(
                      scale=2.0, mode='fan_out', distribution='normal')):
        # c2's MSRAFill is fan_out
        for k in range(num_convs):
            l = Conv2D('fcn{}'.format(k), l, cfg.MRCNN.HEAD_DIM, 3, activation=tf.nn.relu)
        l = Conv2DTranspose('deconv', l, cfg.MRCNN.HEAD_DIM, 2, strides=2, activation=tf.nn.relu)
        l = Conv2D('conv', l, num_category, 1)
    return l


@under_name_scope()
def maskrcnn_loss(mask_logits, fg_labels, fg_target_masks):
    """
    Args:
        mask_logits: #fg x #category xhxw
        fg_labels: #fg, in 1~#class
        fg_target_masks: #fgxhxw, int
    """
    num_fg = tf.size(fg_labels)
    indices = tf.stack([tf.range(num_fg), tf.to_int32(fg_labels) - 1], axis=1)  # #fgx2
    mask_logits = tf.gather_nd(mask_logits, indices)  # #fgxhxw
    mask_probs = tf.sigmoid(mask_logits)

    # add some training visualizations to tensorboard
    with tf.name_scope('mask_viz'):
        viz = tf.concat([fg_target_masks, mask_probs], axis=1)
        viz = tf.expand_dims(viz, 3)
        viz = tf.cast(viz * 255, tf.uint8, name='viz')
        tf.summary.image('mask_truth|pred', viz, max_outputs=10)

    loss = tf.nn.sigmoid_cross_entropy_with_logits(
        labels=fg_target_masks, logits=mask_logits)
    loss = tf.reduce_mean(loss, name='maskrcnn_loss')

    pred_label = mask_probs > 0.5
    truth_label = fg_target_masks > 0.5
    accuracy = tf.reduce_mean(
        tf.to_float(tf.equal(pred_label, truth_label)),
        name='accuracy')
    pos_accuracy = tf.logical_and(
        tf.equal(pred_label, truth_label),
        tf.equal(truth_label, True))
    pos_accuracy = tf.reduce_mean(tf.to_float(pos_accuracy), name='pos_accuracy')
    fg_pixel_ratio = tf.reduce_mean(tf.to_float(truth_label), name='fg_pixel_ratio')

    add_moving_summary(loss, accuracy, fg_pixel_ratio, pos_accuracy)
    return loss


@layer_register(log_shape=True)
def fpn_model(features):
    """
    Args:
        features ([tf.Tensor]): ResNet features c2-c5

    Returns:
        [tf.Tensor]: FPN features p2-p6
    """
    assert len(features) == 4, features
    num_channel = cfg.FPN.NUM_CHANNEL

    def upsample2x(name, x):
        return FixedUnPooling(
            name, x, 2, unpool_mat=np.ones((2, 2), dtype='float32'),
            data_format='channels_first')

        # tf.image.resize is, again, not aligned.
        # with tf.name_scope(name):
        #     shape2d = tf.shape(x)[2:]
        #     x = tf.transpose(x, [0, 2, 3, 1])
        #     x = tf.image.resize_nearest_neighbor(x, shape2d * 2, align_corners=True)
        #     x = tf.transpose(x, [0, 3, 1, 2])
        #     return x

    with argscope(Conv2D, data_format='channels_first',
                  activation=tf.identity, use_bias=True,
                  kernel_initializer=tf.variance_scaling_initializer(scale=1.)):
        lat_2345 = [Conv2D('lateral_1x1_c{}'.format(i + 2), c, num_channel, 1)
                    for i, c in enumerate(features)]
        lat_sum_5432 = []
        for idx, lat in enumerate(lat_2345[::-1]):
            if idx == 0:
                lat_sum_5432.append(lat)
            else:
                lat = lat + upsample2x('upsample_lat{}'.format(6 - idx), lat_sum_5432[-1])
                lat_sum_5432.append(lat)
        p2345 = [Conv2D('posthoc_3x3_p{}'.format(i + 2), c, num_channel, 3)
                 for i, c in enumerate(lat_sum_5432[::-1])]
        p6 = MaxPooling('maxpool_p6', p2345[-1], pool_size=1, strides=2, data_format='channels_first')
        return p2345 + [p6]


@under_name_scope()
def fpn_map_rois_to_levels(boxes):
    """
    Assign boxes to level 2~5.

    Args:
        boxes (nx4):

    Returns:
        [tf.Tensor]: 4 tensors for level 2-5. Each tensor is a vector of indices of boxes in its level.
        [tf.Tensor]: 4 tensors, the gathered boxes in each level.

    Be careful that the returned tensor could be empty.
    """
    sqrtarea = tf.sqrt(tf_area(boxes))
    level = tf.to_int32(tf.floor(
        4 + tf.log(sqrtarea * (1. / 224) + 1e-6) * (1.0 / np.log(2))))

    # RoI levels range from 2~5 (not 6)
    level_ids = [
        tf.where(level <= 2),
        tf.where(tf.equal(level, 3)),   # == is not supported
        tf.where(tf.equal(level, 4)),
        tf.where(level >= 5)]
    level_ids = [tf.reshape(x, [-1], name='roi_level{}_id'.format(i + 2))
                 for i, x in enumerate(level_ids)]
    num_in_levels = [tf.size(x, name='num_roi_level{}'.format(i + 2))
                     for i, x in enumerate(level_ids)]
    add_moving_summary(*num_in_levels)

    level_boxes = [tf.gather(boxes, ids) for ids in level_ids]
    return level_ids, level_boxes


@under_name_scope()
def multilevel_roi_align(features, rcnn_boxes, resolution):
    """
    Args:
        features ([tf.Tensor]): 4 FPN feature level 2-5
        rcnn_boxes (tf.Tensor): nx4 boxes
        resolution (int): output spatial resolution
    Returns:
        NxC x res x res
    """
    assert len(features) == 4, features
    # Reassign rcnn_boxes to levels
    level_ids, level_boxes = fpn_map_rois_to_levels(rcnn_boxes)
    all_rois = []

    # Crop patches from corresponding levels
    for i, boxes, featuremap in zip(itertools.count(), level_boxes, features):
        with tf.name_scope('roi_level{}'.format(i + 2)):
            boxes_on_featuremap = boxes * (1.0 / cfg.FPN.ANCHOR_STRIDES[i])
            all_rois.append(roi_align(featuremap, boxes_on_featuremap, resolution))

    all_rois = tf.concat(all_rois, axis=0)  # NCHW
    # Unshuffle to the original order, to match the original samples
    level_id_perm = tf.concat(level_ids, axis=0)  # A permutation of 1~N
    level_id_invert_perm = tf.invert_permutation(level_id_perm)
    all_rois = tf.gather(all_rois, level_id_invert_perm)
    return all_rois
