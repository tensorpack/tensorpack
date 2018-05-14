#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: train.py

import os
import argparse
import cv2
import shutil
import itertools
import tqdm
import numpy as np
import json
import six
import tensorflow as tf

assert six.PY3, "FasterRCNN requires Python 3!"

from tensorpack import *
from tensorpack.tfutils.summary import add_moving_summary
from tensorpack.tfutils.scope_utils import under_name_scope
from tensorpack.tfutils import optimizer
import tensorpack.utils.viz as tpviz
from tensorpack.utils.gpu import get_nr_gpu


from coco import COCODetection
from basemodel import (
    image_preprocess, pretrained_resnet_c4_backbone, resnet_conv5,
    pretrained_resnet_fpn_backbone)
from model import (
    clip_boxes, decode_bbox_target, encode_bbox_target, crop_and_resize,
    rpn_head, rpn_losses,
    generate_rpn_proposals, sample_fast_rcnn_targets, roi_align,
    fastrcnn_outputs, fastrcnn_losses, fastrcnn_predictions,
    maskrcnn_head, maskrcnn_loss,
    fpn_model, fpn_map_rois_to_levels, fastrcnn_2fc_head)
from data import (
    get_train_dataflow, get_eval_dataflow,
    get_all_anchors, get_all_anchors_fpn)
from viz import (
    draw_annotation, draw_proposal_recall,
    draw_predictions, draw_final_outputs)
from common import print_config
from eval import (
    eval_coco, detect_one_image, print_evaluation_scores, DetectionResult)
import config


def get_batch_factor():
    nr_gpu = get_nr_gpu()
    assert nr_gpu in [1, 2, 4, 8], nr_gpu
    return 8 // nr_gpu


def get_model_output_names():
    ret = ['final_boxes', 'final_probs', 'final_labels']
    if config.MODE_MASK:
        ret.append('final_masks')
    return ret


def get_model():
    if config.MODE_FPN:
        return ResNetFPNModel()
    else:
        return ResNetC4Model()


class DetectionModel(ModelDesc):
    def inputs(self):
        ret = [
            tf.placeholder(tf.float32, (None, None, 3), 'image'),
            tf.placeholder(tf.int32, (None, None, config.NUM_ANCHOR), 'anchor_labels'),
            tf.placeholder(tf.float32, (None, None, config.NUM_ANCHOR, 4), 'anchor_boxes'),
            tf.placeholder(tf.float32, (None, 4), 'gt_boxes'),
            tf.placeholder(tf.int64, (None,), 'gt_labels')]  # all > 0
        if config.MODE_MASK:
            ret.append(
                tf.placeholder(tf.uint8, (None, None, None), 'gt_masks')
            )   # NR_GT x height x width
        return ret

    def preprocess(self, image):
        image = tf.expand_dims(image, 0)
        image = image_preprocess(image, bgr=True)
        return tf.transpose(image, [0, 3, 1, 2])

    @under_name_scope()
    def narrow_to_featuremap(self, featuremap, anchors, anchor_labels, anchor_boxes):
        """
        Args:
            Slice anchors/anchor_labels/anchor_boxes to the spatial size of this featuremap.
            anchors (FS x FS x NA x 4):
            anchor_labels (FS x FS x NA):
            anchor_boxes (FS x FS x NA x 4):
        """
        shape2d = tf.shape(featuremap)[2:]  # h,w
        slice3d = tf.concat([shape2d, [-1]], axis=0)
        slice4d = tf.concat([shape2d, [-1, -1]], axis=0)
        anchors = tf.slice(anchors, [0, 0, 0, 0], slice4d)
        anchor_labels = tf.slice(anchor_labels, [0, 0, 0], slice3d)
        anchor_boxes = tf.slice(anchor_boxes, [0, 0, 0, 0], slice4d)
        return anchors, anchor_labels, anchor_boxes

    def optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=0.003, trainable=False)
        tf.summary.scalar('learning_rate', lr)

        factor = get_batch_factor()
        if factor != 1:
            lr = lr / float(factor)
            opt = tf.train.MomentumOptimizer(lr, 0.9)
            opt = optimizer.AccumGradOptimizer(opt, factor)
        else:
            opt = tf.train.MomentumOptimizer(lr, 0.9)
        return opt

    def fastrcnn_training(self, image,
                          rcnn_labels, fg_rcnn_boxes, gt_boxes_per_fg,
                          rcnn_label_logits, fg_rcnn_box_logits):
        """
        Args:
            image (NCHW):
            rcnn_labels (n): labels for each sampled targets
            fg_rcnn_boxes (fg x 4): proposal boxes for each sampled foreground targets
            gt_boxes_per_fg (fg x 4): matching gt boxes for each sampled foreground targets
            rcnn_label_logits (n): label logits for each sampled targets
            fg_rcnn_box_logits (fg x 4): box logits for each sampled foreground targets
        """

        with tf.name_scope('fg_sample_patch_viz'):
            fg_sampled_patches = crop_and_resize(
                image, fg_rcnn_boxes,
                tf.zeros(tf.shape(fg_rcnn_boxes)[0], dtype=tf.int32), 300)
            fg_sampled_patches = tf.transpose(fg_sampled_patches, [0, 2, 3, 1])
            fg_sampled_patches = tf.reverse(fg_sampled_patches, axis=[-1])  # BGR->RGB
            tf.summary.image('viz', fg_sampled_patches, max_outputs=30)

        encoded_boxes = encode_bbox_target(
            gt_boxes_per_fg, fg_rcnn_boxes) * tf.constant(config.FASTRCNN_BBOX_REG_WEIGHTS)
        fastrcnn_label_loss, fastrcnn_box_loss = fastrcnn_losses(
            rcnn_labels, rcnn_label_logits,
            encoded_boxes,
            fg_rcnn_box_logits)
        return fastrcnn_label_loss, fastrcnn_box_loss

    def fastrcnn_inference(self, image_shape2d,
                           rcnn_boxes, rcnn_label_logits, rcnn_box_logits):
        """
        Args:
            image_shape2d: h, w
            rcnn_boxes (nx4): the proposal boxes
            rcnn_label_logits (n):
            rcnn_box_logits (nx4):

        Returns:
            boxes (mx4):
            labels (m): each >= 1
        """
        label_probs = tf.nn.softmax(rcnn_label_logits, name='fastrcnn_all_probs')  # #proposal x #Class
        anchors = tf.tile(tf.expand_dims(rcnn_boxes, 1), [1, config.NUM_CLASS - 1, 1])   # #proposal x #Cat x 4
        decoded_boxes = decode_bbox_target(
            rcnn_box_logits /
            tf.constant(config.FASTRCNN_BBOX_REG_WEIGHTS), anchors)
        decoded_boxes = clip_boxes(decoded_boxes, image_shape2d, name='fastrcnn_all_boxes')

        # indices: Nx2. Each index into (#proposal, #category)
        pred_indices, final_probs = fastrcnn_predictions(decoded_boxes, label_probs)
        final_probs = tf.identity(final_probs, 'final_probs')
        final_boxes = tf.gather_nd(decoded_boxes, pred_indices, name='final_boxes')
        final_labels = tf.add(pred_indices[:, 1], 1, name='final_labels')
        return final_boxes, final_labels


class ResNetC4Model(DetectionModel):
    def build_graph(self, *inputs):
        is_training = get_current_tower_context().is_training
        if config.MODE_MASK:
            image, anchor_labels, anchor_boxes, gt_boxes, gt_labels, gt_masks = inputs
        else:
            image, anchor_labels, anchor_boxes, gt_boxes, gt_labels = inputs
        image = self.preprocess(image)     # 1CHW

        featuremap = pretrained_resnet_c4_backbone(image, config.RESNET_NUM_BLOCK[:3])
        rpn_label_logits, rpn_box_logits = rpn_head('rpn', featuremap, 1024, config.NUM_ANCHOR)

        fm_anchors, anchor_labels, anchor_boxes = self.narrow_to_featuremap(
            featuremap, get_all_anchors(), anchor_labels, anchor_boxes)
        anchor_boxes_encoded = encode_bbox_target(anchor_boxes, fm_anchors)

        image_shape2d = tf.shape(image)[2:]     # h,w
        pred_boxes_decoded = decode_bbox_target(rpn_box_logits, fm_anchors)  # fHxfWxNAx4, floatbox
        proposal_boxes, proposal_scores = generate_rpn_proposals(
            tf.reshape(pred_boxes_decoded, [-1, 4]),
            tf.reshape(rpn_label_logits, [-1]),
            image_shape2d,
            config.TRAIN_PRE_NMS_TOPK if is_training else config.TEST_PRE_NMS_TOPK,
            config.TRAIN_POST_NMS_TOPK if is_training else config.TEST_POST_NMS_TOPK)

        if is_training:
            # sample proposal boxes in training
            rcnn_boxes, rcnn_labels, fg_inds_wrt_gt = sample_fast_rcnn_targets(
                proposal_boxes, gt_boxes, gt_labels)
        else:
            # The boxes to be used to crop RoIs.
            # Use all proposal boxes in inference
            rcnn_boxes = proposal_boxes

        boxes_on_featuremap = rcnn_boxes * (1.0 / config.ANCHOR_STRIDE)
        roi_resized = roi_align(featuremap, boxes_on_featuremap, 14)

        # HACK to work around https://github.com/tensorflow/tensorflow/issues/14657
        def ff_true():
            feature_fastrcnn = resnet_conv5(roi_resized, config.RESNET_NUM_BLOCK[-1])    # nxcx7x7
            feature_gap = GlobalAvgPooling('gap', feature_fastrcnn, data_format='channels_first')
            fastrcnn_label_logits, fastrcnn_box_logits = fastrcnn_outputs('fastrcnn', feature_gap, config.NUM_CLASS)
            # Return C5 feature to be shared with mask branch
            return feature_fastrcnn, fastrcnn_label_logits, fastrcnn_box_logits

        def ff_false():
            ncls = config.NUM_CLASS
            return tf.zeros([0, 2048, 7, 7]), tf.zeros([0, ncls]), tf.zeros([0, ncls - 1, 4])

        feature_fastrcnn, fastrcnn_label_logits, fastrcnn_box_logits = tf.cond(
            tf.size(boxes_on_featuremap) > 0, ff_true, ff_false)

        if is_training:
            # rpn loss
            rpn_label_loss, rpn_box_loss = rpn_losses(
                anchor_labels, anchor_boxes_encoded, rpn_label_logits, rpn_box_logits)

            # fastrcnn loss
            matched_gt_boxes = tf.gather(gt_boxes, fg_inds_wrt_gt)

            fg_inds_wrt_sample = tf.reshape(tf.where(rcnn_labels > 0), [-1])   # fg inds w.r.t all samples
            fg_sampled_boxes = tf.gather(rcnn_boxes, fg_inds_wrt_sample)
            fg_fastrcnn_box_logits = tf.gather(fastrcnn_box_logits, fg_inds_wrt_sample)

            fastrcnn_label_loss, fastrcnn_box_loss = self.fastrcnn_training(
                image, rcnn_labels, fg_sampled_boxes,
                matched_gt_boxes, fastrcnn_label_logits, fg_fastrcnn_box_logits)

            if config.MODE_MASK:
                # maskrcnn loss
                fg_labels = tf.gather(rcnn_labels, fg_inds_wrt_sample)
                # In training, mask branch shares the same C5 feature.
                fg_feature = tf.gather(feature_fastrcnn, fg_inds_wrt_sample)
                mask_logits = maskrcnn_head('maskrcnn', fg_feature, config.NUM_CLASS)   # #fg x #cat x 14x14

                gt_masks_for_fg = tf.gather(gt_masks, fg_inds_wrt_gt)  # nfg x H x W
                target_masks_for_fg = crop_and_resize(
                    tf.expand_dims(gt_masks_for_fg, 1),
                    fg_sampled_boxes,
                    tf.range(tf.size(fg_inds_wrt_gt)), 14,
                    pad_border=False)  # nfg x 1x14x14
                target_masks_for_fg = tf.squeeze(target_masks_for_fg, 1, 'sampled_fg_mask_targets')
                mrcnn_loss = maskrcnn_loss(mask_logits, fg_labels, target_masks_for_fg)
            else:
                mrcnn_loss = 0.0

            wd_cost = regularize_cost(
                '(?:group1|group2|group3|rpn|fastrcnn|maskrcnn)/.*W',
                l2_regularizer(1e-4), name='wd_cost')

            total_cost = tf.add_n([
                rpn_label_loss, rpn_box_loss,
                fastrcnn_label_loss, fastrcnn_box_loss,
                mrcnn_loss,
                wd_cost], 'total_cost')

            add_moving_summary(total_cost, wd_cost)
            return total_cost
        else:
            final_boxes, final_labels = self.fastrcnn_inference(
                image_shape2d, rcnn_boxes, fastrcnn_label_logits, fastrcnn_box_logits)

            if config.MODE_MASK:
                # HACK to work around https://github.com/tensorflow/tensorflow/issues/14657
                def f1():
                    roi_resized = roi_align(featuremap, final_boxes * (1.0 / config.ANCHOR_STRIDE), 14)
                    feature_maskrcnn = resnet_conv5(roi_resized, config.RESNET_NUM_BLOCK[-1])
                    mask_logits = maskrcnn_head(
                        'maskrcnn', feature_maskrcnn, config.NUM_CLASS)   # #result x #cat x 14x14
                    indices = tf.stack([tf.range(tf.size(final_labels)), tf.to_int32(final_labels) - 1], axis=1)
                    final_mask_logits = tf.gather_nd(mask_logits, indices)   # #resultx14x14
                    return tf.sigmoid(final_mask_logits)

                final_masks = tf.cond(tf.size(final_labels) > 0, f1, lambda: tf.zeros([0, 14, 14]))
                tf.identity(final_masks, name='final_masks')


class ResNetFPNModel(DetectionModel):
    def inputs(self):
        ret = [
            tf.placeholder(tf.float32, (None, None, 3), 'image')]
        num_anchors = len(config.ANCHOR_RATIOS)
        for k in range(len(config.ANCHOR_STRIDES_FPN)):
            ret.extend([
                tf.placeholder(tf.int32, (None, None, num_anchors),
                               'anchor_labels_lvl{}'.format(k + 2)),
                tf.placeholder(tf.float32, (None, None, num_anchors, 4),
                               'anchor_boxes_lvl{}'.format(k + 2))])
        ret.extend([
            tf.placeholder(tf.float32, (None, 4), 'gt_boxes'),
            tf.placeholder(tf.int64, (None,), 'gt_labels')])  # all > 0
        if config.MODE_MASK:
            ret.append(
                tf.placeholder(tf.uint8, (None, None, None), 'gt_masks')
            )   # NR_GT x height x width
        return ret

    def build_graph(self, *inputs):
        num_fpn_level = len(config.ANCHOR_STRIDES_FPN)
        assert len(config.ANCHOR_SIZES) == num_fpn_level
        is_training = get_current_tower_context().is_training
        image = inputs[0]
        input_anchors = inputs[1: 1 + 2 * num_fpn_level]
        multilevel_anchor_labels = input_anchors[0::2]
        multilevel_anchor_boxes = input_anchors[1::2]
        gt_boxes, gt_labels = inputs[11], inputs[12]
        if config.MODE_MASK:
            gt_masks = inputs[-1]

        image = self.preprocess(image)     # 1CHW
        image_shape2d = tf.shape(image)[2:]     # h,w

        c2345 = pretrained_resnet_fpn_backbone(image, config.RESNET_NUM_BLOCK)
        p23456 = fpn_model('fpn', c2345)

        # Multi-Level RPN Proposals
        multilevel_anchors = get_all_anchors_fpn()
        assert len(multilevel_anchors) == num_fpn_level
        multilevel_proposals = []
        rpn_loss_collection = []
        for lvl in range(num_fpn_level):
            rpn_label_logits, rpn_box_logits = rpn_head(
                'rpn', p23456[lvl], config.FPN_NUM_CHANNEL, len(config.ANCHOR_RATIOS))
            with tf.name_scope('FPN_lvl{}'.format(lvl + 2)):
                anchors, anchor_labels, anchor_boxes = \
                    self.narrow_to_featuremap(p23456[lvl], multilevel_anchors[lvl],
                                              multilevel_anchor_labels[lvl],
                                              multilevel_anchor_boxes[lvl])
                anchor_boxes_encoded = encode_bbox_target(anchor_boxes, anchors)
                pred_boxes_decoded = decode_bbox_target(rpn_box_logits, anchors)
                proposal_boxes, proposal_scores = generate_rpn_proposals(
                    tf.reshape(pred_boxes_decoded, [-1, 4]),
                    tf.reshape(rpn_label_logits, [-1]),
                    image_shape2d,
                    config.TRAIN_FPN_NMS_TOPK if is_training else config.TEST_FPN_NMS_TOPK)
                multilevel_proposals.append((proposal_boxes, proposal_scores))
                if is_training:
                    label_loss, box_loss = rpn_losses(
                        anchor_labels, anchor_boxes_encoded,
                        rpn_label_logits, rpn_box_logits)
                    rpn_loss_collection.extend([label_loss, box_loss])

        # merge proposals from multi levels
        proposal_boxes = tf.concat([x[0] for x in multilevel_proposals], axis=0)  # nx4
        proposal_scores = tf.concat([x[1] for x in multilevel_proposals], axis=0)  # n
        proposal_topk = tf.minimum(tf.size(proposal_scores),
                                   config.TRAIN_FPN_NMS_TOPK if is_training else config.TEST_FPN_NMS_TOPK)

        proposal_scores, topk_indices = tf.nn.top_k(proposal_scores, k=proposal_topk, sorted=False)
        proposal_boxes = tf.gather(proposal_boxes, topk_indices)

        if is_training:
            rcnn_boxes, rcnn_labels, fg_inds_wrt_gt = sample_fast_rcnn_targets(
                proposal_boxes, gt_boxes, gt_labels)
        else:
            # The boxes to be used to crop RoIs.
            rcnn_boxes = proposal_boxes

        # Reassign rcnn_boxes to levels
        level_ids, level_boxes = fpn_map_rois_to_levels(rcnn_boxes)
        all_rois = []
        # Crop patches from corresponding levels
        for i, boxes, featuremap in zip(itertools.count(), level_boxes, p23456[:4]):
            with tf.name_scope('roi_level{}'.format(i + 2)):
                boxes_on_featuremap = boxes * (1.0 / config.ANCHOR_STRIDES_FPN[i])
                all_rois.append(roi_align(featuremap, boxes_on_featuremap, 7))
        all_rois = tf.concat(all_rois, axis=0)  # NCHW

        # Unshuffle to the original order, to match the original samples
        level_id_perm = tf.concat(level_ids, axis=0)  # A permutation of 1~N
        level_id_invert_perm = tf.invert_permutation(level_id_perm)
        all_rois = tf.gather(all_rois, level_id_invert_perm)

        fastrcnn_label_logits, fastrcnn_box_logits = fastrcnn_2fc_head(
            'fastrcnn', all_rois, config.FASTRCNN_FC_HEAD_DIM, config.NUM_CLASS)

        if is_training:
            # rpn_losses = ..
            # fastrcnn loss:
            matched_gt_boxes = tf.gather(gt_boxes, fg_inds_wrt_gt)

            fg_inds_wrt_sample = tf.reshape(tf.where(rcnn_labels > 0), [-1])   # fg inds w.r.t all samples
            fg_sampled_boxes = tf.gather(rcnn_boxes, fg_inds_wrt_sample)
            fg_fastrcnn_box_logits = tf.gather(fastrcnn_box_logits, fg_inds_wrt_sample)

            fastrcnn_label_loss, fastrcnn_box_loss = self.fastrcnn_training(
                image, rcnn_labels, fg_sampled_boxes,
                matched_gt_boxes, fastrcnn_label_logits, fg_fastrcnn_box_logits)

            mrcnn_loss = 0.0

            wd_cost = regularize_cost(
                '(?:group1|group2|group3|rpn|fastrcnn|maskrcnn)/.*W',
                l2_regularizer(1e-4), name='wd_cost')

            total_cost = tf.add_n(rpn_loss_collection + [
                fastrcnn_label_loss, fastrcnn_box_loss,
                mrcnn_loss, wd_cost], 'total_cost')

            add_moving_summary(total_cost, wd_cost)
            return total_cost
        else:
            final_boxes, final_labels = self.fastrcnn_inference(
                image_shape2d, rcnn_boxes, fastrcnn_label_logits, fastrcnn_box_logits)


def visualize(model_path, nr_visualize=50, output_dir='output'):
    df = get_train_dataflow()   # we don't visualize mask stuff
    df.reset_state()

    pred = OfflinePredictor(PredictConfig(
        model=ResNetC4Model(),
        session_init=get_model_loader(model_path),
        input_names=['image', 'gt_boxes', 'gt_labels'],
        output_names=[
            'generate_rpn_proposals/boxes',
            'generate_rpn_proposals/probs',
            'fastrcnn_all_probs',
            'final_boxes',
            'final_probs',
            'final_labels',
        ]))

    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    utils.fs.mkdir_p(output_dir)
    with tqdm.tqdm(total=nr_visualize) as pbar:
        for idx, dp in itertools.islice(enumerate(df.get_data()), nr_visualize):
            img, _, _, gt_boxes, gt_labels = dp

            rpn_boxes, rpn_scores, all_probs, \
                final_boxes, final_probs, final_labels = pred(img, gt_boxes, gt_labels)

            # draw groundtruth boxes
            gt_viz = draw_annotation(img, gt_boxes, gt_labels)
            # draw best proposals for each groundtruth, to show recall
            proposal_viz, good_proposals_ind = draw_proposal_recall(img, rpn_boxes, rpn_scores, gt_boxes)
            # draw the scores for the above proposals
            score_viz = draw_predictions(img, rpn_boxes[good_proposals_ind], all_probs[good_proposals_ind])

            results = [DetectionResult(*args) for args in
                       zip(final_boxes, final_probs, final_labels,
                           [None] * len(final_labels))]
            final_viz = draw_final_outputs(img, results)

            viz = tpviz.stack_patches([
                gt_viz, proposal_viz,
                score_viz, final_viz], 2, 2)

            if os.environ.get('DISPLAY', None):
                tpviz.interactive_imshow(viz)
            cv2.imwrite("{}/{:03d}.png".format(output_dir, idx), viz)
            pbar.update()


def offline_evaluate(pred_func, output_file):
    df = get_eval_dataflow()
    all_results = eval_coco(
        df, lambda img: detect_one_image(img, pred_func))
    with open(output_file, 'w') as f:
        json.dump(all_results, f)
    print_evaluation_scores(output_file)


def predict(pred_func, input_file):
    img = cv2.imread(input_file, cv2.IMREAD_COLOR)
    results = detect_one_image(img, pred_func)
    final = draw_final_outputs(img, results)
    viz = np.concatenate((img, final), axis=1)
    tpviz.interactive_imshow(viz)


class EvalCallback(Callback):
    def _setup_graph(self):
        self.pred = self.trainer.get_predictor(
            ['image'], get_model_output_names())
        self.df = get_eval_dataflow()

    def _before_train(self):
        EVAL_TIMES = 5  # eval 5 times during training
        interval = self.trainer.max_epoch // (EVAL_TIMES + 1)
        self.epochs_to_eval = set([interval * k for k in range(1, EVAL_TIMES)])
        self.epochs_to_eval.add(self.trainer.max_epoch)

    def _eval(self):
        all_results = eval_coco(self.df, lambda img: detect_one_image(img, self.pred))
        output_file = os.path.join(
            logger.get_logger_dir(), 'outputs{}.json'.format(self.global_step))
        with open(output_file, 'w') as f:
            json.dump(all_results, f)
        scores = print_evaluation_scores(output_file)
        for k, v in scores.items():
            self.trainer.monitors.put_scalar(k, v)

    def _trigger_epoch(self):
        if self.epoch_num in self.epochs_to_eval:
            self._eval()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use. Default to all availalbe ones')
    parser.add_argument('--load', help='load model for evaluation or training')
    parser.add_argument('--logdir', help='log directory', default='train_log/maskrcnn')
    parser.add_argument('--datadir', help='override config.BASEDIR')
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--evaluate', help="Run evaluation on COCO. "
                                           "This option is the path to the output json evaluation file")
    parser.add_argument('--predict', help="Run prediction on a given image. "
                                          "This argument is the path to the input image file")
    args = parser.parse_args()
    if args.datadir:
        config.BASEDIR = args.datadir

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if args.visualize or args.evaluate or args.predict:
        # autotune is too slow for inference
        os.environ['TF_CUDNN_USE_AUTOTUNE'] = '0'

        assert args.load
        print_config()

        if args.predict or args.visualize:
            config.RESULT_SCORE_THRESH = config.RESULT_SCORE_THRESH_VIS

        if args.visualize:
            visualize(args.load)
        else:
            pred = OfflinePredictor(PredictConfig(
                model=get_model(),
                session_init=get_model_loader(args.load),
                input_names=['image'],
                output_names=get_model_output_names()))
            if args.evaluate:
                assert args.evaluate.endswith('.json')
                offline_evaluate(pred, args.evaluate)
            elif args.predict:
                COCODetection(config.BASEDIR, 'val2014')   # Only to load the class names into caches
                predict(pred, args.predict)
    else:
        logger.set_logger_dir(args.logdir, 'd')
        print_config()
        factor = get_batch_factor()
        stepnum = config.STEPS_PER_EPOCH

        # warmup is step based, lr is epoch based
        warmup_schedule = [(0, config.BASE_LR / 3), (config.WARMUP * factor, config.BASE_LR)]
        warmup_end_epoch = config.WARMUP * factor * 1. / stepnum
        lr_schedule = [(int(np.ceil(warmup_end_epoch)), warmup_schedule[-1][1])]
        for idx, steps in enumerate(config.LR_SCHEDULE[:-1]):
            mult = 0.1 ** (idx + 1)
            lr_schedule.append(
                (steps * factor // stepnum, config.BASE_LR * mult))

        cfg = TrainConfig(
            model=get_model(),
            data=QueueInput(get_train_dataflow(add_mask=config.MODE_MASK)),
            callbacks=[
                PeriodicCallback(
                    ModelSaver(max_to_keep=10, keep_checkpoint_every_n_hours=1),
                    every_k_epochs=20),
                # linear warmup
                ScheduledHyperParamSetter(
                    'learning_rate', warmup_schedule, interp='linear', step_based=True),
                ScheduledHyperParamSetter('learning_rate', lr_schedule),
                EvalCallback(),
                GPUUtilizationTracker(),
                EstimatedTimeLeft(),
            ],
            steps_per_epoch=stepnum,
            max_epoch=config.LR_SCHEDULE[-1] * factor // stepnum,
            session_init=get_model_loader(args.load) if args.load else None,
        )
        trainer = SyncMultiGPUTrainerReplicated(get_nr_gpu())
        launch_train_with_config(cfg, trainer)
