#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: imagenet-resnet.py

import argparse
import os

from tensorpack import QueueInput, TFDatasetInput, logger
from tensorpack.callbacks import *
from tensorpack.dataflow import FakeData
from tensorpack.models import *
from tensorpack.tfutils import argscope, SmartInit
from tensorpack.train import SyncMultiGPUTrainerReplicated, TrainConfig, launch_train_with_config
from tensorpack.utils.gpu import get_num_gpu

from imagenet_utils import ImageNetModel, eval_classification, get_imagenet_dataflow, get_imagenet_tfdata
import resnet_model
from resnet_model import preact_group, resnet_backbone, resnet_group


class Model(ImageNetModel):
    def __init__(self, depth, mode='resnet'):
        self.mode = mode
        basicblock = getattr(resnet_model, mode + '_basicblock', None)
        bottleneck = getattr(resnet_model, mode + '_bottleneck', None)
        self.num_blocks, self.block_func = {
            18: ([2, 2, 2, 2], basicblock),
            34: ([3, 4, 6, 3], basicblock),
            50: ([3, 4, 6, 3], bottleneck),
            101: ([3, 4, 23, 3], bottleneck),
            152: ([3, 8, 36, 3], bottleneck)
        }[depth]
        assert self.block_func is not None, \
            "(mode={}, depth={}) not implemented!".format(mode, depth)

    def get_logits(self, image):
        with argscope([Conv2D, MaxPooling, GlobalAvgPooling, BatchNorm], data_format=self.data_format):
            return resnet_backbone(
                image, self.num_blocks,
                preact_group if self.mode == 'preact' else resnet_group, self.block_func)


def get_config(model):
    nr_tower = max(get_num_gpu(), 1)
    assert args.batch % nr_tower == 0
    batch = args.batch // nr_tower

    logger.info("Running on {} towers. Batch size per tower: {}".format(nr_tower, batch))
    if batch < 32 or batch > 64:
        logger.warn("Batch size per tower not in [32, 64]. This probably will lead to worse accuracy than reported.")
    if args.fake:
        data = QueueInput(FakeData(
            [[batch, 224, 224, 3], [batch]], 1000, random=False, dtype='uint8'))
        callbacks = []
    else:
        if args.symbolic:
            data = TFDatasetInput(get_imagenet_tfdata(args.data, 'train', batch))
        else:
            data = QueueInput(get_imagenet_dataflow(args.data, 'train', batch))

        START_LR = 0.1
        BASE_LR = START_LR * (args.batch / 256.0)
        callbacks = [
            ModelSaver(),
            EstimatedTimeLeft(),
            ScheduledHyperParamSetter(
                'learning_rate', [
                    (0, min(START_LR, BASE_LR)), (30, BASE_LR * 1e-1), (60, BASE_LR * 1e-2),
                    (90, BASE_LR * 1e-3), (100, BASE_LR * 1e-4)]),
        ]
        if BASE_LR > START_LR:
            callbacks.append(
                ScheduledHyperParamSetter(
                    'learning_rate', [(0, START_LR), (5, BASE_LR)], interp='linear'))

        infs = [ClassificationError('wrong-top1', 'val-error-top1'),
                ClassificationError('wrong-top5', 'val-error-top5')]
        dataset_val = get_imagenet_dataflow(args.data, 'val', batch)
        if nr_tower == 1:
            # single-GPU inference with queue prefetch
            callbacks.append(InferenceRunner(QueueInput(dataset_val), infs))
        else:
            # multi-GPU inference (with mandatory queue prefetch)
            callbacks.append(DataParallelInferenceRunner(
                dataset_val, infs, list(range(nr_tower))))

    if get_num_gpu() > 0:
        callbacks.append(GPUUtilizationTracker())

    return TrainConfig(
        model=model,
        data=data,
        callbacks=callbacks,
        steps_per_epoch=100 if args.fake else 1281167 // args.batch,
        max_epoch=105,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # generic:
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use. Default to use all available ones')
    parser.add_argument('--eval', action='store_true', help='run offline evaluation instead of training')
    parser.add_argument('--load', help='load a model for training or evaluation')

    # data:
    parser.add_argument('--data', help='ILSVRC dataset dir')
    parser.add_argument('--fake', help='use FakeData to debug or benchmark this model', action='store_true')
    parser.add_argument('--symbolic', help='use symbolic data loader', action='store_true')

    # model:
    parser.add_argument('--data-format', help='the image data layout used by the model',
                        default='NCHW', choices=['NCHW', 'NHWC'])
    parser.add_argument('-d', '--depth', help='ResNet depth',
                        type=int, default=50, choices=[18, 34, 50, 101, 152])
    parser.add_argument('--weight-decay-norm', action='store_true',
                        help="apply weight decay on normalization layers (gamma & beta)."
                             "This is used in torch/pytorch, and slightly "
                             "improves validation accuracy of large models.")
    parser.add_argument('--batch', default=256, type=int,
                        help="total batch size. "
                        "Note that it's best to keep per-GPU batch size in [32, 64] to obtain the best accuracy."
                        "Pretrained models listed in README were trained with batch=32x8.")
    parser.add_argument('--mode', choices=['resnet', 'preact', 'se', 'resnext32x4d'],
                        help='variants of resnet to use', default='resnet')
    args = parser.parse_args()

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    model = Model(args.depth, args.mode)
    model.data_format = args.data_format
    if args.weight_decay_norm:
        model.weight_decay_pattern = ".*/W|.*/gamma|.*/beta"

    if args.eval:
        batch = 128    # something that can run on one gpu
        ds = get_imagenet_dataflow(args.data, 'val', batch)
        eval_classification(model, SmartInit(args.load), ds)
    else:
        if args.fake:
            logger.set_logger_dir(os.path.join('train_log', 'tmp'), 'd')
        else:
            logger.set_logger_dir(
                os.path.join('train_log',
                             'imagenet-{}-d{}-batch{}'.format(
                                 args.mode, args.depth, args.batch)))

        config = get_config(model)
        config.session_init = SmartInit(args.load)
        trainer = SyncMultiGPUTrainerReplicated(max(get_num_gpu(), 1))
        launch_train_with_config(config, trainer)
