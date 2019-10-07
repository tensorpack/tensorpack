#!/bin/bash -ev
# File: run-tests.sh

mkdir -p "$TENSORPACK_DATASET"
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
python -m unittest tensorpack.callbacks.param_test
python -m unittest tensorpack.tfutils.unit_tests
python -m unittest tensorpack.dataflow.imgaug.imgaug_test
python -m unittest tensorpack.models.models_test

# use pyarrow after we organize the serializers.
# TENSORPACK_SERIALIZE=pyarrow python ...
python -m unittest tensorpack.dataflow.serialize_test

# e2e tests
python -m unittest discover -v
