#!/usr/bin/env python
# -*- coding: utf-8 -*-

## export.py
import argparse
import tensorflow as tf

from tensorpack import *
from tensorpack.tfutils.export import ModelExporter
from modeling.generalized_rcnn import ResNetC4Model, ResNetFPNModel
from config import config as cfg
from dataset import DatasetRegistry, register_coco
from config import finalize_configs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', help="A list of KEY=VALUE to overwrite those defined in config.py",
                            nargs='+')
    parser.add_argument('--load', help='load a model for evaluation.', required=True)
    parser.add_argument('--compact', help='if you want to save a model to .pb')
    parser.add_argument('--serving', help='if you want to save a model to serving file')
    args = parser.parse_args()


    # Update config
    if args.config:
        cfg.update_args(args.config)
    MODEL = ResNetFPNModel() if cfg.MODE_FPN else ResNetC4Model()
    finalize_configs(is_training=False)


    predcfg = PredictConfig(
                model=MODEL,
                session_init=get_model_loader(args.load),
                input_names=MODEL.get_inference_tensor_names()[0], # image
                output_names=MODEL.get_inference_tensor_names()[1]) # box, score, label, mask

    # Make file
    # I use to Tensorflow-gpu 1.8.0. when i use optimize variable is 'True', not work. so, i use optimzie variable is 'False'.
    # I don't know different tensorflow-gpu version.
    if args.compact:
            ModelExporter(predcfg).export_compact(args.compact, optimize=False)
    elif args.serving:
            ModelExporter(predcfg).export_serving(args.serving, optimize=False)



