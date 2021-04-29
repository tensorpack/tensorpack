import tensorflow as tf

from tensorpack.tfutils import get_current_tower_context

from config import config as cfg
from utils.box_ops import pairwise_iou, area as tf_area
from .model_box import clip_boxes
from .model_frcnn import BoxProposals, FastRCNNHead, fastrcnn_outputs


class CascadeRCNNHead(object):
    def __init__(self, proposals,
                 roi_func, fastrcnn_head_func, gt_targets, image_shape2d,
                 num_categories):
        """
        Args:
            proposals: BoxProposals
            roi_func (boxes -> features): a function to crop features with rois
            fastrcnn_head_func (features -> features): the fastrcnn head to apply on the cropped features
            gt_targets (gt_boxes, gt_labels):
        """
        for k, v in locals().items():
            if k != 'self':
                setattr(self, k, v)
        self.gt_boxes, self.gt_labels = gt_targets
        del self.gt_targets

        self.num_cascade_stages = len(cfg.CASCADE.IOUS)

        self.training = get_current_tower_context().is_training
        if self.training:
            @tf.custom_gradient
            def scale_gradient(x):
                return x, lambda dy: dy * (1.0 / self.num_cascade_stages)
            self.scale_gradient = scale_gradient
        else:
            self.scale_gradient = tf.identity

        ious = cfg.CASCADE.IOUS
        # It's unclear how to do >3 stages, so it does not make sense to implement them
        assert self.num_cascade_stages == 3, "Only 3-stage cascade was implemented!"
        with tf.variable_scope('cascade_rcnn_stage1'):
            H1, B1 = self.run_head(self.proposals, 0)

        with tf.variable_scope('cascade_rcnn_stage2'):
            B1_proposal = self.match_box_with_gt(B1, ious[1])
            H2, B2 = self.run_head(B1_proposal, 1)

        with tf.variable_scope('cascade_rcnn_stage3'):
            B2_proposal = self.match_box_with_gt(B2, ious[2])
            H3, B3 = self.run_head(B2_proposal, 2)
        self._cascade_boxes = [B1, B2, B3]
        self._heads = [H1, H2, H3]

    def run_head(self, proposals, stage):
        """
        Args:
            proposals: BoxProposals
            stage: 0, 1, 2

        Returns:
            FastRCNNHead
            Nx4, updated boxes
        """
        reg_weights = tf.constant(cfg.CASCADE.BBOX_REG_WEIGHTS[stage], dtype=tf.float32)
        pooled_feature = self.roi_func(proposals.boxes)  # N,C,S,S
        pooled_feature = self.scale_gradient(pooled_feature)
        head_feature = self.fastrcnn_head_func('head', pooled_feature)
        label_logits, box_logits = fastrcnn_outputs(
            'outputs', head_feature, self.num_categories, class_agnostic_regression=True)
        head = FastRCNNHead(proposals, box_logits, label_logits, self.gt_boxes, reg_weights)

        refined_boxes = head.decoded_output_boxes_class_agnostic()
        refined_boxes = clip_boxes(refined_boxes, self.image_shape2d)
        if self.training:
            refined_boxes = tf.boolean_mask(refined_boxes, tf_area(refined_boxes) > 0)
        return head, tf.stop_gradient(refined_boxes, name='output_boxes')

    def match_box_with_gt(self, boxes, iou_threshold):
        """
        Args:
            boxes: Nx4
        Returns:
            BoxProposals
        """
        if self.training:
            with tf.name_scope('match_box_with_gt_{}'.format(iou_threshold)):
                iou = pairwise_iou(boxes, self.gt_boxes)  # NxM
                max_iou_per_box = tf.reduce_max(iou, axis=1)  # N
                best_iou_ind = tf.cond(tf.shape(iou)[1] > 0,
                                       lambda: tf.argmax(iou, axis=1),   # #proposal, each in 0~m-1
                                       lambda: tf.zeros(tf.shape(iou)[0], dtype=tf.int64))
                labels_per_box = tf.gather(self.gt_labels, best_iou_ind)
                fg_mask = max_iou_per_box >= iou_threshold
                fg_inds_wrt_gt = tf.boolean_mask(best_iou_ind, fg_mask)
                labels_per_box = tf.stop_gradient(labels_per_box * tf.cast(fg_mask, tf.int64))
                return BoxProposals(boxes, labels_per_box, fg_inds_wrt_gt)
        else:
            return BoxProposals(boxes)

    def losses(self):
        ret = []
        for idx, head in enumerate(self._heads):
            with tf.name_scope('cascade_loss_stage{}'.format(idx + 1)):
                ret.extend(head.losses())
        return ret

    def decoded_output_boxes(self):
        """
        Returns:
            Nx#classx4
        """
        ret = self._cascade_boxes[-1]
        ret = tf.expand_dims(ret, 1)     # class-agnostic
        return tf.tile(ret, [1, self.num_categories + 1, 1])

    def output_scores(self, name=None):
        """
        Returns:
            Nx#class
        """
        scores = [head.output_scores('cascade_scores_stage{}'.format(idx + 1))
                  for idx, head in enumerate(self._heads)]
        return tf.multiply(tf.add_n(scores), (1.0 / self.num_cascade_stages), name=name)
