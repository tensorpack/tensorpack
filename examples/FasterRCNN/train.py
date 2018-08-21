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
from concurrent.futures import ThreadPoolExecutor
try:
    import horovod.tensorflow as hvd
except ImportError:
    pass

assert six.PY3, "FasterRCNN requires Python 3!"

from tensorpack import *
from tensorpack.tfutils.summary import add_moving_summary
from tensorpack.tfutils import optimizer
from tensorpack.tfutils.common import get_tf_version_tuple
import tensorpack.utils.viz as tpviz

from coco import COCODetection
from basemodel import (
    image_preprocess, resnet_c4_backbone, resnet_conv5,
    resnet_fpn_backbone)

import model_frcnn
import model_mrcnn
from model_frcnn import (
    sample_fast_rcnn_targets,
    fastrcnn_outputs, fastrcnn_predictions, FastRCNNHead)
from model_mrcnn import maskrcnn_upXconv_head, maskrcnn_loss
from model_rpn import rpn_head, rpn_losses, generate_rpn_proposals
from model_fpn import (
    fpn_model, multilevel_roi_align,
    multilevel_rpn_losses, generate_fpn_proposals)
from model_box import (
    clip_boxes, crop_and_resize, roi_align, RPNAnchors)

from data import (
    get_train_dataflow, get_eval_dataflow,
    get_all_anchors, get_all_anchors_fpn)
from viz import (
    draw_annotation, draw_proposal_recall,
    draw_predictions, draw_final_outputs)
from eval import (
    eval_coco, detect_one_image, print_evaluation_scores, DetectionResult)
from config import finalize_configs, config as cfg


class DetectionModel(ModelDesc):
    def preprocess(self, image):
        image = tf.expand_dims(image, 0)
        image = image_preprocess(image, bgr=True)
        return tf.transpose(image, [0, 3, 1, 2])

    def optimizer(self):
        lr = tf.get_variable('learning_rate', initializer=0.003, trainable=False)
        tf.summary.scalar('learning_rate-summary', lr)

        # The learning rate is set for 8 GPUs, and we use trainers with average=False.
        lr = lr / 8.
        opt = tf.train.MomentumOptimizer(lr, 0.9)
        if cfg.TRAIN.NUM_GPUS < 8:
            opt = optimizer.AccumGradOptimizer(opt, 8 // cfg.TRAIN.NUM_GPUS)
        return opt

    def fastrcnn_inference(self, image_shape2d, fastrcnn_head):
        """
        Args:
            image_shape2d: h, w
            fastrcnn_head (FastRCNNHead):

        Returns:
            boxes (mx4):
            labels (m): each >= 1
        """
        decoded_boxes = fastrcnn_head.decoded_output_boxes()
        decoded_boxes = clip_boxes(decoded_boxes, image_shape2d, name='fastrcnn_all_boxes')
        label_probs = fastrcnn_head.output_scores(name='fastrcnn_all_probs')

        # indices: Nx2. Each index into (#box, #class)
        pred_indices, final_probs = fastrcnn_predictions(decoded_boxes, label_probs)
        final_probs = tf.identity(final_probs, 'final_probs')
        final_boxes = tf.gather_nd(decoded_boxes, pred_indices, name='final_boxes')
        final_labels = tf.gather(pred_indices, 1, axis=1, name='final_labels')
        return final_boxes, final_labels

    def get_inference_tensor_names(self):
        """
        Returns two lists of tensor names to be used to create an inference callable.

        Returns:
            [str]: input names
            [str]: output names
        """
        out = ['final_boxes', 'final_probs', 'final_labels']
        if cfg.MODE_MASK:
            out.append('final_masks')
        return ['image'], out


class ResNetC4Model(DetectionModel):
    def inputs(self):
        ret = [
            tf.placeholder(tf.float32, (None, None, 3), 'image'),
            tf.placeholder(tf.int32, (None, None, cfg.RPN.NUM_ANCHOR), 'anchor_labels'),
            tf.placeholder(tf.float32, (None, None, cfg.RPN.NUM_ANCHOR, 4), 'anchor_boxes'),
            tf.placeholder(tf.float32, (None, 4), 'gt_boxes'),
            tf.placeholder(tf.int64, (None,), 'gt_labels')]  # all > 0
        if cfg.MODE_MASK:
            ret.append(
                tf.placeholder(tf.uint8, (None, None, None), 'gt_masks')
            )   # NR_GT x height x width
        return ret

    def build_graph(self, *inputs):
        inputs = dict(zip(self.input_names, inputs))
        is_training = get_current_tower_context().is_training
        image = self.preprocess(inputs['image'])     # 1CHW

        featuremap = resnet_c4_backbone(image, cfg.BACKBONE.RESNET_NUM_BLOCK[:3])
        rpn_label_logits, rpn_box_logits = rpn_head('rpn', featuremap, cfg.RPN.HEAD_DIM, cfg.RPN.NUM_ANCHOR)

        anchors = RPNAnchors(get_all_anchors(), inputs['anchor_labels'], inputs['anchor_boxes'])
        anchors = anchors.narrow_to(featuremap)

        image_shape2d = tf.shape(image)[2:]     # h,w
        pred_boxes_decoded = anchors.decode_logits(rpn_box_logits)  # fHxfWxNAx4, floatbox
        proposal_boxes, proposal_scores = generate_rpn_proposals(
            tf.reshape(pred_boxes_decoded, [-1, 4]),
            tf.reshape(rpn_label_logits, [-1]),
            image_shape2d,
            cfg.RPN.TRAIN_PRE_NMS_TOPK if is_training else cfg.RPN.TEST_PRE_NMS_TOPK,
            cfg.RPN.TRAIN_POST_NMS_TOPK if is_training else cfg.RPN.TEST_POST_NMS_TOPK)

        gt_boxes, gt_labels = inputs['gt_boxes'], inputs['gt_labels']
        if is_training:
            # sample proposal boxes in training
            rcnn_boxes, rcnn_labels, fg_inds_wrt_gt = sample_fast_rcnn_targets(
                proposal_boxes, gt_boxes, gt_labels)
            matched_gt_boxes = tf.gather(gt_boxes, fg_inds_wrt_gt, name='gt_boxes_per_fg_proposal')
        else:
            # The boxes to be used to crop RoIs.
            # Use all proposal boxes in inference
            rcnn_boxes = proposal_boxes
            rcnn_labels, matched_gt_boxes = None, None

        boxes_on_featuremap = rcnn_boxes * (1.0 / cfg.RPN.ANCHOR_STRIDE)
        roi_resized = roi_align(featuremap, boxes_on_featuremap, 14)

        feature_fastrcnn = resnet_conv5(roi_resized, cfg.BACKBONE.RESNET_NUM_BLOCK[-1])    # nxcx7x7
        # Keep C5 feature to be shared with mask branch
        feature_gap = GlobalAvgPooling('gap', feature_fastrcnn, data_format='channels_first')
        fastrcnn_label_logits, fastrcnn_box_logits = fastrcnn_outputs('fastrcnn', feature_gap, cfg.DATA.NUM_CLASS)

        fastrcnn_head = FastRCNNHead(rcnn_boxes, fastrcnn_box_logits, fastrcnn_label_logits,
                                     tf.constant(cfg.FRCNN.BBOX_REG_WEIGHTS, dtype=tf.float32),
                                     rcnn_labels, matched_gt_boxes)

        if is_training:
            # rpn loss
            rpn_label_loss, rpn_box_loss = rpn_losses(
                anchors.gt_labels, anchors.encoded_gt_boxes(), rpn_label_logits, rpn_box_logits)

            # fastrcnn loss
            fastrcnn_label_loss, fastrcnn_box_loss = fastrcnn_head.losses()

            if cfg.MODE_MASK:
                # maskrcnn loss
                # In training, mask branch shares the same C5 feature.
                fg_feature = tf.gather(feature_fastrcnn, fastrcnn_head.fg_inds_in_inputs())
                mask_logits = maskrcnn_upXconv_head(
                    'maskrcnn', fg_feature, cfg.DATA.NUM_CATEGORY, num_convs=0)   # #fg x #cat x 14x14

                target_masks_for_fg = crop_and_resize(
                    tf.expand_dims(inputs['gt_masks'], 1),
                    fastrcnn_head.fg_input_boxes(),
                    fg_inds_wrt_gt, 14,
                    pad_border=False)  # nfg x 1x14x14
                target_masks_for_fg = tf.squeeze(target_masks_for_fg, 1, 'sampled_fg_mask_targets')
                mrcnn_loss = maskrcnn_loss(mask_logits, fastrcnn_head.fg_labels(), target_masks_for_fg)
            else:
                mrcnn_loss = 0.0

            wd_cost = regularize_cost(
                '.*/W', l2_regularizer(cfg.TRAIN.WEIGHT_DECAY), name='wd_cost')

            total_cost = tf.add_n([
                rpn_label_loss, rpn_box_loss,
                fastrcnn_label_loss, fastrcnn_box_loss,
                mrcnn_loss, wd_cost], 'total_cost')

            add_moving_summary(total_cost, wd_cost)
            return total_cost
        else:
            final_boxes, final_labels = self.fastrcnn_inference(image_shape2d, fastrcnn_head)

            if cfg.MODE_MASK:
                roi_resized = roi_align(featuremap, final_boxes * (1.0 / cfg.RPN.ANCHOR_STRIDE), 14)
                feature_maskrcnn = resnet_conv5(roi_resized, cfg.BACKBONE.RESNET_NUM_BLOCK[-1])
                mask_logits = maskrcnn_upXconv_head(
                    'maskrcnn', feature_maskrcnn, cfg.DATA.NUM_CATEGORY, 0)   # #result x #cat x 14x14
                indices = tf.stack([tf.range(tf.size(final_labels)), tf.to_int32(final_labels) - 1], axis=1)
                final_mask_logits = tf.gather_nd(mask_logits, indices)   # #resultx14x14
                tf.sigmoid(final_mask_logits, name='final_masks')


class ResNetFPNModel(DetectionModel):

    def inputs(self):
        ret = [
            tf.placeholder(tf.float32, (None, None, 3), 'image')]
        num_anchors = len(cfg.RPN.ANCHOR_RATIOS)
        for k in range(len(cfg.FPN.ANCHOR_STRIDES)):
            ret.extend([
                tf.placeholder(tf.int32, (None, None, num_anchors),
                               'anchor_labels_lvl{}'.format(k + 2)),
                tf.placeholder(tf.float32, (None, None, num_anchors, 4),
                               'anchor_boxes_lvl{}'.format(k + 2))])
        ret.extend([
            tf.placeholder(tf.float32, (None, 4), 'gt_boxes'),
            tf.placeholder(tf.int64, (None,), 'gt_labels')])  # all > 0
        if cfg.MODE_MASK:
            ret.append(
                tf.placeholder(tf.uint8, (None, None, None), 'gt_masks')
            )   # NR_GT x height x width
        return ret

    def slice_feature_and_anchors(self, image_shape2d, p23456, anchors):
        for i, stride in enumerate(cfg.FPN.ANCHOR_STRIDES):
            with tf.name_scope('FPN_slice_lvl{}'.format(i)):
                if i < 3:
                    # Images are padded for p5, which are too large for p2-p4.
                    # This seems to have no effect on mAP.
                    pi = p23456[i]
                    target_shape = tf.to_int32(tf.ceil(tf.to_float(image_shape2d) * (1.0 / stride)))
                    p23456[i] = tf.slice(pi, [0, 0, 0, 0],
                                         tf.concat([[-1, -1], target_shape], axis=0))
                    p23456[i].set_shape([1, pi.shape[1], None, None])

                anchors[i] = anchors[i].narrow_to(p23456[i])

    def build_graph(self, *inputs):
        inputs = dict(zip(self.input_names, inputs))
        num_fpn_level = len(cfg.FPN.ANCHOR_STRIDES)
        assert len(cfg.RPN.ANCHOR_SIZES) == num_fpn_level
        is_training = get_current_tower_context().is_training

        all_anchors_fpn = get_all_anchors_fpn()
        multilevel_anchors = [RPNAnchors(
            all_anchors_fpn[i],
            inputs['anchor_labels_lvl{}'.format(i + 2)],
            inputs['anchor_boxes_lvl{}'.format(i + 2)]) for i in range(len(all_anchors_fpn))]

        image = self.preprocess(inputs['image'])     # 1CHW
        image_shape2d = tf.shape(image)[2:]     # h,w

        c2345 = resnet_fpn_backbone(image, cfg.BACKBONE.RESNET_NUM_BLOCK)
        p23456 = fpn_model('fpn', c2345)
        self.slice_feature_and_anchors(image_shape2d, p23456, multilevel_anchors)

        # Multi-Level RPN Proposals
        rpn_outputs = [rpn_head('rpn', pi, cfg.FPN.NUM_CHANNEL, len(cfg.RPN.ANCHOR_RATIOS))
                       for pi in p23456]
        multilevel_label_logits = [k[0] for k in rpn_outputs]
        multilevel_box_logits = [k[1] for k in rpn_outputs]

        proposal_boxes, proposal_scores = generate_fpn_proposals(
            multilevel_anchors, multilevel_label_logits,
            multilevel_box_logits, image_shape2d)

        gt_boxes, gt_labels = inputs['gt_boxes'], inputs['gt_labels']
        if is_training:
            rcnn_boxes, rcnn_labels, fg_inds_wrt_gt = sample_fast_rcnn_targets(
                proposal_boxes, gt_boxes, gt_labels)
            matched_gt_boxes = tf.gather(gt_boxes, fg_inds_wrt_gt)
        else:
            # The boxes to be used to crop RoIs.
            rcnn_boxes = proposal_boxes
            rcnn_labels, matched_gt_boxes = None, None

        roi_feature_fastrcnn = multilevel_roi_align(p23456[:4], rcnn_boxes, 7)

        fastrcnn_head_func = getattr(model_frcnn, cfg.FPN.FRCNN_HEAD_FUNC)
        fastrcnn_label_logits, fastrcnn_box_logits = fastrcnn_head_func(
            'fastrcnn', roi_feature_fastrcnn, cfg.DATA.NUM_CLASS)
        fastrcnn_head = FastRCNNHead(rcnn_boxes, fastrcnn_box_logits, fastrcnn_label_logits,
                                     tf.constant(cfg.FRCNN.BBOX_REG_WEIGHTS, dtype=tf.float32),
                                     rcnn_labels, matched_gt_boxes)

        if is_training:
            # rpn loss:
            rpn_label_loss, rpn_box_loss = multilevel_rpn_losses(
                multilevel_anchors, multilevel_label_logits, multilevel_box_logits)

            fastrcnn_label_loss, fastrcnn_box_loss = fastrcnn_head.losses()

            if cfg.MODE_MASK:
                # maskrcnn loss
                roi_feature_maskrcnn = multilevel_roi_align(
                    p23456[:4], fastrcnn_head.fg_input_boxes(), 14,
                    name_scope='multilevel_roi_align_mask')
                maskrcnn_head_func = getattr(model_mrcnn, cfg.FPN.MRCNN_HEAD_FUNC)
                mask_logits = maskrcnn_head_func(
                    'maskrcnn', roi_feature_maskrcnn, cfg.DATA.NUM_CATEGORY)   # #fg x #cat x 28 x 28

                target_masks_for_fg = crop_and_resize(
                    tf.expand_dims(inputs['gt_masks'], 1),
                    fastrcnn_head.fg_input_boxes(),
                    fg_inds_wrt_gt, 28,
                    pad_border=False)  # fg x 1x28x28
                target_masks_for_fg = tf.squeeze(target_masks_for_fg, 1, 'sampled_fg_mask_targets')
                mrcnn_loss = maskrcnn_loss(mask_logits, fastrcnn_head.fg_labels(), target_masks_for_fg)
            else:
                mrcnn_loss = 0.0

            wd_cost = regularize_cost(
                '.*/W', l2_regularizer(cfg.TRAIN.WEIGHT_DECAY), name='wd_cost')

            total_cost = tf.add_n([rpn_label_loss, rpn_box_loss,
                                   fastrcnn_label_loss, fastrcnn_box_loss,
                                   mrcnn_loss, wd_cost], 'total_cost')

            add_moving_summary(total_cost, wd_cost)
            return total_cost
        else:
            final_boxes, final_labels = self.fastrcnn_inference(image_shape2d, fastrcnn_head)
            if cfg.MODE_MASK:
                # Cascade inference needs roi transform with refined boxes.
                roi_feature_maskrcnn = multilevel_roi_align(p23456[:4], final_boxes, 14)
                maskrcnn_head_func = getattr(model_mrcnn, cfg.FPN.MRCNN_HEAD_FUNC)
                mask_logits = maskrcnn_head_func(
                    'maskrcnn', roi_feature_maskrcnn, cfg.DATA.NUM_CATEGORY)   # #fg x #cat x 28 x 28
                indices = tf.stack([tf.range(tf.size(final_labels)), tf.to_int32(final_labels) - 1], axis=1)
                final_mask_logits = tf.gather_nd(mask_logits, indices)   # #resultx28x28
                tf.sigmoid(final_mask_logits, name='final_masks')


def visualize(model, model_path, nr_visualize=100, output_dir='output'):
    """
    Visualize some intermediate results (proposals, raw predictions) inside the pipeline.
    """
    df = get_train_dataflow()   # we don't visualize mask stuff
    df.reset_state()

    pred = OfflinePredictor(PredictConfig(
        model=model,
        session_init=get_model_loader(model_path),
        input_names=['image', 'gt_boxes', 'gt_labels'],
        output_names=[
            'generate_{}_proposals/boxes'.format('fpn' if cfg.MODE_FPN else 'rpn'),
            'generate_{}_proposals/probs'.format('fpn' if cfg.MODE_FPN else 'rpn'),
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
            img = dp[0]
            if cfg.MODE_MASK:
                gt_boxes, gt_labels, gt_masks = dp[-3:]
            else:
                gt_boxes, gt_labels = dp[-2:]

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
    """
    A callback that runs COCO evaluation once a while.
    It supports multi-GPU evaluation if TRAINER=='replicated' and single-GPU evaluation if TRAINER=='horovod'
    """

    def __init__(self, in_names, out_names):
        self._in_names, self._out_names = in_names, out_names

    def _setup_graph(self):
        num_gpu = cfg.TRAIN.NUM_GPUS
        # Use two predictor threads per GPU to get better throughput
        self.num_predictor = 1 if cfg.TRAINER == 'horovod' else num_gpu * 2
        self.predictors = [self._build_coco_predictor(k % num_gpu) for k in range(self.num_predictor)]
        self.dataflows = [get_eval_dataflow(shard=k, num_shards=self.num_predictor)
                          for k in range(self.num_predictor)]

    def _build_coco_predictor(self, idx):
        graph_func = self.trainer.get_predictor(self._in_names, self._out_names, device=idx)
        return lambda img: detect_one_image(img, graph_func)

    def _before_train(self):
        num_eval = cfg.TRAIN.NUM_EVALS
        interval = max(self.trainer.max_epoch // (num_eval + 1), 1)
        self.epochs_to_eval = set([interval * k for k in range(1, num_eval + 1)])
        self.epochs_to_eval.add(self.trainer.max_epoch)
        if len(self.epochs_to_eval) < 15:
            logger.info("[EvalCallback] Will evaluate at epoch " + str(sorted(self.epochs_to_eval)))
        else:
            if cfg.TRAINER == 'horovod':
                logger.warn("[EvalCallback] Evaluation is single-GPU only and quite slow under horovod mode.")
            logger.info("[EvalCallback] Will evaluate every {} epochs".format(interval))

    def _eval(self):
        with ThreadPoolExecutor(max_workers=self.num_predictor, thread_name_prefix='EvalWorker') as executor, \
                tqdm.tqdm(total=sum([df.size() for df in self.dataflows])) as pbar:
            futures = []
            for dataflow, pred in zip(self.dataflows, self.predictors):
                futures.append(executor.submit(eval_coco, dataflow, pred, pbar))
            all_results = list(itertools.chain(*[fut.result() for fut in futures]))

        output_file = os.path.join(
            logger.get_logger_dir(), 'outputs{}.json'.format(self.global_step))
        with open(output_file, 'w') as f:
            json.dump(all_results, f)
        try:
            scores = print_evaluation_scores(output_file)
            for k, v in scores.items():
                self.trainer.monitors.put_scalar(k, v)
        except Exception:
            logger.exception("Exception in COCO evaluation.")

    def _trigger_epoch(self):
        if self.epoch_num in self.epochs_to_eval:
            self._eval()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', help='load a model for evaluation or training. Can overwrite BACKBONE.WEIGHTS')
    parser.add_argument('--logdir', help='log directory', default='train_log/maskrcnn')
    parser.add_argument('--visualize', action='store_true', help='visualize intermediate results')
    parser.add_argument('--evaluate', help="Run evaluation on COCO. "
                                           "This argument is the path to the output json evaluation file")
    parser.add_argument('--predict', help="Run prediction on a given image. "
                                          "This argument is the path to the input image file")
    parser.add_argument('--config', help="A list of KEY=VALUE to overwrite those defined in config.py",
                        nargs='+')

    if get_tf_version_tuple() < (1, 6):
        # https://github.com/tensorflow/tensorflow/issues/14657
        logger.warn("TF<1.6 has a bug which may lead to crash in FasterRCNN training if you're unlucky.")

    args = parser.parse_args()
    if args.config:
        cfg.update_args(args.config)

    MODEL = ResNetFPNModel() if cfg.MODE_FPN else ResNetC4Model()

    if args.visualize or args.evaluate or args.predict:
        assert args.load
        finalize_configs(is_training=False)

        if args.predict or args.visualize:
            cfg.TEST.RESULT_SCORE_THRESH = cfg.TEST.RESULT_SCORE_THRESH_VIS

        if args.visualize:
            visualize(MODEL, args.load)
        else:
            pred = OfflinePredictor(PredictConfig(
                model=MODEL,
                session_init=get_model_loader(args.load),
                input_names=MODEL.get_inference_tensor_names()[0],
                output_names=MODEL.get_inference_tensor_names()[1]))
            if args.evaluate:
                assert args.evaluate.endswith('.json'), args.evaluate
                offline_evaluate(pred, args.evaluate)
            elif args.predict:
                COCODetection(cfg.DATA.BASEDIR, 'val2014')   # Only to load the class names into caches
                predict(pred, args.predict)
    else:
        is_horovod = cfg.TRAINER == 'horovod'
        if is_horovod:
            hvd.init()
            logger.info("Horovod Rank={}, Size={}".format(hvd.rank(), hvd.size()))

        if not is_horovod or hvd.rank() == 0:
            logger.set_logger_dir(args.logdir, 'd')

        finalize_configs(is_training=True)
        stepnum = cfg.TRAIN.STEPS_PER_EPOCH

        # warmup is step based, lr is epoch based
        init_lr = cfg.TRAIN.BASE_LR * 0.33 * min(8. / cfg.TRAIN.NUM_GPUS, 1.)
        warmup_schedule = [(0, init_lr), (cfg.TRAIN.WARMUP, cfg.TRAIN.BASE_LR)]
        warmup_end_epoch = cfg.TRAIN.WARMUP * 1. / stepnum
        lr_schedule = [(int(np.ceil(warmup_end_epoch)), cfg.TRAIN.BASE_LR)]

        factor = 8. / cfg.TRAIN.NUM_GPUS
        for idx, steps in enumerate(cfg.TRAIN.LR_SCHEDULE[:-1]):
            mult = 0.1 ** (idx + 1)
            lr_schedule.append(
                (steps * factor // stepnum, cfg.TRAIN.BASE_LR * mult))
        logger.info("Warm Up Schedule (steps, value): " + str(warmup_schedule))
        logger.info("LR Schedule (epochs, value): " + str(lr_schedule))

        callbacks = [
            PeriodicCallback(
                ModelSaver(max_to_keep=10, keep_checkpoint_every_n_hours=1),
                every_k_epochs=20),
            # linear warmup
            ScheduledHyperParamSetter(
                'learning_rate', warmup_schedule, interp='linear', step_based=True),
            ScheduledHyperParamSetter('learning_rate', lr_schedule),
            EvalCallback(*MODEL.get_inference_tensor_names()),
            PeakMemoryTracker(),
            EstimatedTimeLeft(median=True),
            SessionRunTimeout(60000).set_chief_only(True),   # 1 minute timeout
        ]
        if not is_horovod:
            callbacks.append(GPUUtilizationTracker())

        if args.load:
            session_init = get_model_loader(args.load)
        else:
            session_init = get_model_loader(cfg.BACKBONE.WEIGHTS) if cfg.BACKBONE.WEIGHTS else None

        traincfg = TrainConfig(
            model=MODEL,
            data=QueueInput(get_train_dataflow()),
            callbacks=callbacks,
            steps_per_epoch=stepnum,
            max_epoch=cfg.TRAIN.LR_SCHEDULE[-1] * factor // stepnum,
            session_init=session_init,
        )
        if is_horovod:
            trainer = HorovodTrainer(average=False)
        else:
            # nccl mode has better speed than cpu mode
            trainer = SyncMultiGPUTrainerReplicated(cfg.TRAIN.NUM_GPUS, average=False, mode='nccl')
        launch_train_with_config(traincfg, trainer)
