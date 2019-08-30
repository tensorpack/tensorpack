#!/bin/bash -ev
# File: run-tests.sh

DIR=$(dirname $0)
cd $DIR

export TF_CPP_MIN_LOG_LEVEL=2
export TF_CPP_MIN_VLOG_LEVEL=2
# test import (#471)
python -c 'from tensorpack.dataflow import imgaug'
# Check that these private names can be imported because tensorpack is using them
python -c "from tensorflow.python.client.session import _FetchHandler"
python -c "from tensorflow.python.training.monitored_session import _HookedSession"
python -c "import tensorflow as tf; tf.Operation._add_control_input"

# run tests
python -m tensorpack.callbacks.param_test
python -m tensorpack.tfutils.unit_tests
python -m unittest tensorpack.dataflow.imgaug._test
# use pyarrow after we organize the serializers.
# TENSORPACK_SERIALIZE=pyarrow python test_serializer.py
TENSORPACK_SERIALIZE=msgpack python test_serializer.py

python -m unittest discover -v
