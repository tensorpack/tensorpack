#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: checkpoint-manipulate.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>


import numpy as np
from tensorpack.tfutils.varmanip import dump_chkpt_vars
from tensorpack.utils import logger
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('checkpoint')
parser.add_argument('--dump', help='dump to an npy file')
parser.add_argument('--shell', action='store_true', help='start a shell with the params')
args = parser.parse_args()

if args.checkpoint.endswith('.npy'):
    params = np.load(args.checkpoint).item()
else:
    params = dump_chkpt_vars(args.checkpoint)
logger.info("Variables in the checkpoint:")
logger.info(str(params.keys()))
if args.dump:
    np.save(args.dump, params)
if args.shell:
    import IPython as IP
    IP.embed(config=IP.terminal.ipapp.load_default_config())
