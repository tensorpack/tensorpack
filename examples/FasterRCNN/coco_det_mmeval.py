# -*- coding: utf-8 -*-
# File: coco_det_mmeval.py

# Copyright (c) OpenMMLab. All rights reserved.

# flake8: noqa

import argparse
import numpy as np
import pycocotools.mask as cocomask
import tensorflow as tf
import tqdm
from config import config as cfg
from config import finalize_configs
from data import get_eval_dataflow
from dataset import DatasetRegistry, register_coco
from eval import predict_image
from modeling.generalized_rcnn import ResNetC4Model, ResNetFPNModel
from mpi4py import MPI
from tensorpack.predict import MultiTowerOfflinePredictor, PredictConfig
from tensorpack.tfutils import SmartInit, get_tf_version_tuple
from tensorpack.utils import logger

from mmeval.metrics import COCODetectionMetric  # type: ignore


def xywh2xyxy(box):
    return [box[0], box[1], box[0] + box[2], box[1] + box[3]]


def do_mmeval_evaluate(pred_config, dataset):
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    dataflow = get_eval_dataflow(dataset, shard=rank, num_shards=size)
    dataflow.reset_state()
    if rank == 0:
        dataflow = tqdm.tqdm(dataflow)

    graph_func = MultiTowerOfflinePredictor(pred_config, [
        rank,
    ]).get_predictors()[0]

    coco_dataset = DatasetRegistry.get(dataset)
    coco_api = coco_dataset.coco

    coco_metric = COCODetectionMetric(
        ann_file=coco_dataset.annotation_file,
        metric=['bbox', 'segm'] if cfg.MODE_MASK else ['bbox'],
        proposal_nums=[1, 10, 100],
        dist_backend='mpi4py')
    coco_metric.dataset_meta = {
        'CLASSES': [cat['name'] for cat in coco_api.cats.values()]
    }

    for img, img_id in dataflow:
        pred_results = predict_image(img, graph_func)
        pred = {
            'bboxes': [],
            'scores': [],
            'labels': [],
        }
        if cfg.MODE_MASK:
            pred['masks'] = []
            pred['mask_scores'] = []

        for r in pred_results:
            pred['bboxes'].append(r.box)
            pred['scores'].append(r.score)
            pred['labels'].append(r.class_id - 1)

            # # also append segmentation to results
            if r.mask is not None and cfg.MODE_MASK:
                rle = cocomask.encode(np.array(r.mask[:, :, None],
                                               order='F'))[0]
                rle['counts'] = rle['counts'].decode('ascii')
                pred['masks'].append(rle)
                pred['mask_scores'].append(r.score)

        for k, v in pred.items():
            pred[k] = np.asarray(v)

        pred['img_id'] = img_id
        coco_metric.add_predictions([pred])

    return coco_metric.compute()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--load', help='load a model for evaluation.', required=True)
    parser.add_argument(
        '--config',
        help='A list of KEY=VALUE to overwrite those defined in config.py',
        nargs='+')

    args = parser.parse_args()
    if args.config:
        cfg.update_args(args.config)
    register_coco(cfg.DATA.BASEDIR)  # add COCO datasets to the registry

    MODEL = ResNetFPNModel() if cfg.MODE_FPN else ResNetC4Model()

    if not tf.test.is_gpu_available():
        from tensorflow.python.framework import test_util
        assert get_tf_version_tuple() >= (1, 7) and test_util.IsMklEnabled(), \
            'Inference requires either GPU support or MKL support!'
    assert args.load
    finalize_configs(is_training=False)

    predcfg = PredictConfig(
        model=MODEL,
        session_init=SmartInit(args.load),
        input_names=MODEL.get_inference_tensor_names()[0],
        output_names=MODEL.get_inference_tensor_names()[1])

    for dataset in cfg.DATA.VAL:
        logger.info(f'Evaluating {dataset} ...')
        do_mmeval_evaluate(predcfg, dataset)