#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: train.py

import os
import argparse
import cv2
import shutil
import itertools
import tqdm
import math
import numpy as np
import json
import tensorflow as tf

os.environ['TENSORPACK_TRAIN_API'] = 'v2'   # will become default soon
from tensorpack import *
from tensorpack.tfutils.summary import add_moving_summary
from tensorpack.tfutils import optimizer
import tensorpack.utils.viz as tpviz
from tensorpack.utils.gpu import get_nr_gpu


from coco import COCODetection
from basemodel import (
    image_preprocess, pretrained_resnet_conv4, resnet_conv5)
from model import (
    clip_boxes, decode_bbox_target, encode_bbox_target,
    rpn_head, rpn_losses,
    generate_rpn_proposals, sample_fast_rcnn_targets, roi_align,
    fastrcnn_head, fastrcnn_losses, fastrcnn_predictions)
from data import (
    get_train_dataflow, get_eval_dataflow,
    get_all_anchors)
from viz import (
    draw_annotation, draw_proposal_recall,
    draw_predictions, draw_final_outputs)
from common import print_config
from eval import (
    eval_on_dataflow, detect_one_image, print_evaluation_scores)
import config


def get_batch_factor():
    nr_gpu = get_nr_gpu()
    assert nr_gpu in [1, 2, 4, 8], nr_gpu
    return 8 // nr_gpu


class Model(ModelDesc):
    def _get_inputs(self):
        return [
            InputDesc(tf.float32, (None, None, 3), 'image'),
            InputDesc(tf.int32, (None, None, config.NUM_ANCHOR), 'anchor_labels'),
            InputDesc(tf.float32, (None, None, config.NUM_ANCHOR, 4), 'anchor_boxes'),
            InputDesc(tf.float32, (None, 4), 'gt_boxes'),
            InputDesc(tf.int64, (None,), 'gt_labels'),  # all > 0
        ]

    def _preprocess(self, image):
        image = tf.expand_dims(image, 0)
        image = image_preprocess(image, bgr=True)
        return tf.transpose(image, [0, 3, 1, 2])

    def _get_anchors(self, image):
        """
        Returns:
            FSxFSxNAx4 anchors,
        """
        # FSxFSxNAx4 (FS=MAX_SIZE//ANCHOR_STRIDE)
        with tf.name_scope('anchors'):
            all_anchors = tf.constant(get_all_anchors(), name='all_anchors', dtype=tf.float32)
            fm_anchors = tf.slice(
                all_anchors, [0, 0, 0, 0], tf.stack([
                    tf.shape(image)[0] // config.ANCHOR_STRIDE,
                    tf.shape(image)[1] // config.ANCHOR_STRIDE,
                    -1, -1]), name='fm_anchors')
            return fm_anchors

    def _build_graph(self, inputs):
        is_training = get_current_tower_context().is_training
        image, anchor_labels, anchor_boxes, gt_boxes, gt_labels = inputs
        fm_anchors = self._get_anchors(image)
        image = self._preprocess(image)
        image_shape2d = tf.shape(image)[2:]

        anchor_boxes_encoded = encode_bbox_target(anchor_boxes, fm_anchors)
        featuremap = pretrained_resnet_conv4(image, config.RESNET_NUM_BLOCK[:3])
        rpn_label_logits, rpn_box_logits = rpn_head('rpn', featuremap, 1024, config.NUM_ANCHOR)

        decoded_boxes = decode_bbox_target(rpn_box_logits, fm_anchors)  # fHxfWxNAx4, floatbox
        proposal_boxes, proposal_scores = generate_rpn_proposals(
            tf.reshape(decoded_boxes, [-1, 4]),
            tf.reshape(rpn_label_logits, [-1]),
            image_shape2d)

        if is_training:
            # sample proposal boxes in training
            rcnn_sampled_boxes, rcnn_labels, fg_inds_wrt_gt = sample_fast_rcnn_targets(
                proposal_boxes, gt_boxes, gt_labels)
            boxes_on_featuremap = rcnn_sampled_boxes * (1.0 / config.ANCHOR_STRIDE)
        else:
            # use all proposal boxes in inference
            boxes_on_featuremap = proposal_boxes * (1.0 / config.ANCHOR_STRIDE)

        roi_resized = roi_align(featuremap, boxes_on_featuremap, 14)
        feature_fastrcnn = resnet_conv5(roi_resized, config.RESNET_NUM_BLOCK[-1])    # nxcx7x7
        fastrcnn_label_logits, fastrcnn_box_logits = fastrcnn_head('fastrcnn', feature_fastrcnn, config.NUM_CLASS)

        if is_training:
            # rpn loss
            rpn_label_loss, rpn_box_loss = rpn_losses(
                anchor_labels, anchor_boxes_encoded, rpn_label_logits, rpn_box_logits)

            # fastrcnn loss
            fg_inds_wrt_sample = tf.reshape(tf.where(rcnn_labels > 0), [-1])   # fg inds w.r.t all samples
            fg_sampled_boxes = tf.gather(rcnn_sampled_boxes, fg_inds_wrt_sample)

            matched_gt_boxes = tf.gather(gt_boxes, fg_inds_wrt_gt)
            encoded_boxes = encode_bbox_target(
                matched_gt_boxes,
                fg_sampled_boxes) * tf.constant(config.FASTRCNN_BBOX_REG_WEIGHTS)
            fastrcnn_label_loss, fastrcnn_box_loss = fastrcnn_losses(
                rcnn_labels, fastrcnn_label_logits,
                encoded_boxes,
                tf.gather(fastrcnn_box_logits, fg_inds_wrt_sample))

            wd_cost = regularize_cost(
                '(?:group1|group2|group3|rpn|fastrcnn)/.*W',
                l2_regularizer(1e-4), name='wd_cost')

            self.cost = tf.add_n([
                rpn_label_loss, rpn_box_loss,
                fastrcnn_label_loss, fastrcnn_box_loss,
                wd_cost], 'total_cost')

            add_moving_summary(self.cost, wd_cost)
        else:
            label_probs = tf.nn.softmax(fastrcnn_label_logits, name='fastrcnn_all_probs')  # #proposal x #Class
            anchors = tf.tile(tf.expand_dims(proposal_boxes, 1), [1, config.NUM_CLASS - 1, 1])   # #proposal x #Cat x 4
            decoded_boxes = decode_bbox_target(
                fastrcnn_box_logits /
                tf.constant(config.FASTRCNN_BBOX_REG_WEIGHTS), anchors)
            decoded_boxes = clip_boxes(decoded_boxes, image_shape2d, name='fastrcnn_all_boxes')

            # indices: Nx2. Each index into (#proposal, #category)
            pred_indices, final_probs = fastrcnn_predictions(decoded_boxes, label_probs)
            final_probs = tf.identity(final_probs, 'final_probs')
            tf.gather_nd(decoded_boxes, pred_indices, name='final_boxes')
            tf.add(pred_indices[:, 1], 1, name='final_labels')

    def _get_optimizer(self):
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


def visualize(model_path, nr_visualize=50, output_dir='output'):
    pred = OfflinePredictor(PredictConfig(
        model=Model(),
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
    df = get_train_dataflow()
    df.reset_state()

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

            results = [DetectionResult(*args) for args in zip(final_labels, final_boxes, final_probs)]
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
    all_results = eval_on_dataflow(
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
            ['image'],
            ['final_boxes', 'final_probs', 'final_labels'])
        self.df = get_eval_dataflow()

    def _before_train(self):
        EVAL_TIMES = 5  # eval 5 times during training
        interval = self.trainer.max_epoch // (EVAL_TIMES + 1)
        self.epochs_to_eval = set([interval * k for k in range(1, EVAL_TIMES)])
        self.epochs_to_eval.add(self.trainer.max_epoch)

    def _eval(self):
        all_results = eval_on_dataflow(self.df, lambda img: detect_one_image(img, self.pred))
        output_file = os.path.join(
            logger.get_logger_dir(), 'outputs{}.json'.format(self.global_step))
        with open(output_file, 'w') as f:
            json.dump(all_results, f)
        print_evaluation_scores(output_file)

    def _trigger_epoch(self):
        if self.epoch_num in self.epochs_to_eval:
            self._eval()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--load', help='load model')
    parser.add_argument('--logdir', help='logdir', default='train_log/fastrcnn')
    parser.add_argument('--datadir', help='override config.BASEDIR')
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--evaluate', help='path to the output json eval file')
    parser.add_argument('--predict', help='path to the input image file')
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
        if args.visualize:
            visualize(args.load)
        else:
            pred = OfflinePredictor(PredictConfig(
                model=Model(),
                session_init=get_model_loader(args.load),
                input_names=['image'],
                output_names=[
                    'final_boxes',
                    'final_probs',
                    'final_labels',
                ]))
            if args.evaluate:
                assert args.evaluate.endswith('.json')
                offline_evaluate(pred, args.evaluate)
            elif args.predict:
                COCODetection(config.BASEDIR, 'train2014')   # to load the class names into caches
                predict(pred, args.predict)
    else:
        logger.set_logger_dir(args.logdir)
        print_config()
        stepnum = 300
        warmup_epoch = max(math.ceil(500.0 / stepnum), 5)
        factor = get_batch_factor()

        cfg = TrainConfig(
            model=Model(),
            dataflow=get_train_dataflow(),
            callbacks=[
                PeriodicTrigger(ModelSaver(), every_k_epochs=5),
                # linear warmup
                ScheduledHyperParamSetter(
                    'learning_rate',
                    [(0, 3e-3), (warmup_epoch * factor, 1e-2)], interp='linear'),
                # step decay
                ScheduledHyperParamSetter(
                    'learning_rate',
                    [(warmup_epoch * factor, 1e-2),
                     (150000 * factor // stepnum, 1e-3),
                     (210000 * factor // stepnum, 1e-4)]),
                EvalCallback(),
                GPUUtilizationTracker(),
            ],
            steps_per_epoch=stepnum,
            max_epoch=230000 * factor // stepnum,
            session_init=get_model_loader(args.load) if args.load else None,
        )
        trainer = SyncMultiGPUTrainerReplicated(get_nr_gpu())
        launch_train_with_config(cfg, trainer)
