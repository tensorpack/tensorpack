#!/bin/bash -e
# File: run-tests.sh
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>

DIR=$(dirname $0)
cd $DIR

export TF_CPP_MIN_LOG_LEVEL=2
# test import (#471)
python -c 'from tensorpack.dataflow.imgaug import transform'

python -m unittest discover -v
# python -m tensorpack.models._test
# segfault for no reason (https://travis-ci.org/ppwwyyxx/tensorpack/jobs/217702985)

# python ../tensorpack/user_ops/test-recv-op.py
