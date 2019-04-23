#
# Copyright (c) 2019 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: EPL-2.0
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys

from tensorflow.core.framework import graph_pb2
from tensorflow.python.platform import app
from tensorflow.python.platform import flags as flags_lib
from tensorflow.python.platform import gfile
from google.protobuf import text_format

import common_flags

flags = flags_lib
FLAGS = flags.FLAGS


def main(unused_args):
    if not gfile.Exists(FLAGS.input):
        print("Input graph file '" + FLAGS.input + "' does not exist!")
        return -1

    tf_graph = graph_pb2.GraphDef()
    # TODO(intel-tf): Enabling user to work with both binary and text format.
    mode = "rb" if FLAGS.input_binary else "r"
    with gfile.Open(FLAGS.input, mode) as f:
        data = f.read()
        if FLAGS.input_binary:
            tf_graph.ParseFromString(data)
        else:
            text_format.Merge(data, tf_graph)

    for node in tf_graph.node:
        new_input = []
        for i in node.input:
            if i[0] == '^':
                node.input.remove(i)

    # TODO(intel-tf): Enabling user to work with both binary and text format.
    mode = "wb" if FLAGS.output_binary else "w"
    f = gfile.FastGFile(FLAGS.output, mode)
    if FLAGS.output_binary:
        f.write(tf_graph.SerializeToString())
    else:
        f.write(str(tf_graph))

    return 0


if __name__ == "__main__":
    app.run()
