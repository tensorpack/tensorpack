#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: serve-data.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

import argparse
import imp
from tensorpack.dataflow import serve_data

parser = argparse.ArgumentParser()
parser.add_argument(dest='config')
parser.add_argument('-p', '--port', help='port', type=int, required=True)
args = parser.parse_args()

get_config_func = imp.load_source('config_script', args.config).get_config
config = get_config_func()

ds = config.dataset
serve_data(ds, "tcp://*:{}".format(args.port))
