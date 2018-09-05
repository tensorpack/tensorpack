# -*- coding: utf-8 -*-
# File: export.py

"""
A collection of functions to ease the process of exporting
a model for production.

"""

import tensorflow as tf

from ..tfutils.common import get_tensors_by_names
from ..tfutils.tower import PredictTowerContext
from ..input_source import PlaceholderInput
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile
from tensorflow.python.tools import optimize_for_inference_lib

__all__ = ['ServingExporter', 'MobileExporter']


class MobileExporter(object):
    """Convert a checkpoint to a frozen and pruned graph."""

    def __init__(self, config):
        self.config = config

    def export(self, export_graph_file, dtype=tf.float32):
        """Apply all graph modifications and write final graph to disk.

        Args:
            export_graph_file (str): path to final local of the graph
            dtype (TYPE, optional): The placeholder data type, or
                a list that specifies one value per input node name.
        """
        self.graph = self.config._maybe_create_graph()
        with self.graph.as_default():
            input = PlaceholderInput()
            input.setup(self.config.inputs_desc)
            with PredictTowerContext(''):
                self.config.tower_func(*input.get_input_tensors())

            input_tensors = get_tensors_by_names(self.config.input_names)
            output_tensors = get_tensors_by_names(self.config.output_names)

            self.config.session_init._setup_graph()
            # we cannot use "self.config.session_creator.create_session()" here since it finalizes the graph
            sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
            self.config.session_init._run_init(sess)

            # freeze variables to constants
            frozen_graph_def = graph_util.convert_variables_to_constants(
                sess,
                self.graph.as_graph_def(),
                [n.name[:-2] for n in output_tensors],
                variable_names_whitelist=None,
                variable_names_blacklist=None)

            # prune unused nodes from graph
            mobile_graph_def = optimize_for_inference_lib.optimize_for_inference(
                frozen_graph_def,
                [n.name[:-2] for n in input_tensors],
                [n.name[:-2] for n in output_tensors],
                dtype.as_datatype_enum,
                False)

            with gfile.FastGFile(export_graph_file, "wb") as f:
                f.write(mobile_graph_def.SerializeToString())


class ServingExporter(object):
    """Converts and checkpoint to a servable for TensorFlow Serving"""

    def __init__(self, config):
        """Initialise the export process.

        Args:
            config (PredictConfig): the config to use.
        """

        self.config = config

    def export(self, export_path,
               tags=[tf.saved_model.tag_constants.SERVING],
               signature_name='prediction_pipeline'):
        """
        Use SavedModelBuilder to export a trained model without tensorpack dependency.

        Remarks:
            This produces
                variables/       # output from the vanilla Saver
                    variables.data-?????-of-?????
                    variables.index
                saved_model.pb   # a `SavedModel` protobuf

            Currently, we only support a single signature, which is the general PredictSignatureDef:
            https://github.com/tensorflow/serving/blob/master/tensorflow_serving/g3doc/signature_defs.md

        Args:
            export_path (str): path for export directory
            tags (list): list of user specified tags
            signature_name (str): name of signature for prediction
        """

        self.graph = self.config._maybe_create_graph()
        with self.graph.as_default():
            input = PlaceholderInput()
            input.setup(self.config.inputs_desc)
            with PredictTowerContext(''):
                self.config.tower_func(*input.get_input_tensors())

            input_tensors = get_tensors_by_names(self.config.input_names)
            inputs_signatures = {t.name: tf.saved_model.utils.build_tensor_info(t) for t in input_tensors}
            output_tensors = get_tensors_by_names(self.config.output_names)
            outputs_signatures = {t.name: tf.saved_model.utils.build_tensor_info(t) for t in output_tensors}

            self.config.session_init._setup_graph()
            # we cannot use "self.config.session_creator.create_session()" here since it finalizes the graph
            sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
            self.config.session_init._run_init(sess)

            builder = tf.saved_model.builder.SavedModelBuilder(export_path)

            prediction_signature = tf.saved_model.signature_def_utils.build_signature_def(
                inputs=inputs_signatures,
                outputs=outputs_signatures,
                method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)

            builder.add_meta_graph_and_variables(
                sess, tags,
                signature_def_map={signature_name: prediction_signature})
            builder.save()
