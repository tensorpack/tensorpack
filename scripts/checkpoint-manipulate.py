#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: checkpoint-manipulate.py


import argparse
import numpy as np

from tensorpack.tfutils.varmanip import load_chkpt_vars
from tensorpack.utils import logger

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('--dump', help='dump to an npz file')
    parser.add_argument('--shell', action='store_true', help='start a shell with the params')
    args = parser.parse_args()

    if args.model.endswith('.npy'):
        params = np.load(args.model, encoding='latin1').item()
    elif args.model.endswith('.npz'):
        params = dict(np.load(args.model))
    else:
        params = load_chkpt_vars(args.model)
    logger.info("Variables in the model:")
    logger.info(str(params.keys()))

    if args.dump:
        assert args.dump.endswith('.npz'), args.dump
        np.savez(args.dump, **params)

    if args.shell:
        # params is a dict. play with it
        import IPython as IP
        IP.embed(config=IP.terminal.ipapp.load_default_config())
