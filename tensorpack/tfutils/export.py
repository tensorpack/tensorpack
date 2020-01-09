# -*- coding: utf-8 -*-
# File: export.py

"""
A collection of functions to ease the process of exporting
a model for production.

"""

import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile
from tensorflow.python.tools import optimize_for_inference_lib

from ..compat import is_tfv2, tfv1
from ..input_source import PlaceholderInput
from ..tfutils.common import get_tensors_by_names, get_tf_version_tuple
from ..tfutils.tower import PredictTowerContext
from ..utils import logger

__all__ = ['ModelExporter']


class ModelExporter(object):
    """Export models for inference."""

    def __init__(self, config):
        """Initialise the export process.

        Args:
            config (PredictConfig): the config to use.
                The graph will be built with the tower function defined by this `PredictConfig`.
                Then the input / output names will be used to export models for inference.
        """
        super(ModelExporter, self).__init__()
        self.config = config

    def export_compact(self, filename, optimize=True, toco_compatible=False):
        """Create a self-contained inference-only graph and write final graph (in pb format) to disk.

        Args:
            filename (str): path to the output graph
            optimize (bool): whether to use TensorFlow's `optimize_for_inference`
                to prune and optimize the graph. This does not work on all types of graphs.
            toco_compatible (bool): See TensorFlow's
                `optimize_for_inference
                <https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/tools/optimize_for_inference.py>`_
                for details. Only available after TF 1.8.
        """
        if toco_compatible:
            assert optimize, "toco_compatible is only effective when optimize=True!"
        self.graph = self.config._maybe_create_graph()
        with self.graph.as_default():
            input = PlaceholderInput()
            input.setup(self.config.input_signature)
            with PredictTowerContext(''):
                self.config.tower_func(*input.get_input_tensors())

            input_tensors = get_tensors_by_names(self.config.input_names)
            output_tensors = get_tensors_by_names(self.config.output_names)

            self.config.session_init._setup_graph()
            # we cannot use "self.config.session_creator.create_session()" here since it finalizes the graph
            sess = tfv1.Session(config=tfv1.ConfigProto(allow_soft_placement=True))
            self.config.session_init._run_init(sess)

            dtypes = [n.dtype for n in input_tensors]

            # freeze variables to constants
            frozen_graph_def = graph_util.convert_variables_to_constants(
                sess,
                self.graph.as_graph_def(),
                [n.name[:-2] for n in output_tensors],
                variable_names_whitelist=None,
                variable_names_blacklist=None)

            # prune unused nodes from graph
            if optimize:
                toco_args = () if get_tf_version_tuple() < (1, 8) else (toco_compatible, )
                frozen_graph_def = optimize_for_inference_lib.optimize_for_inference(
                    frozen_graph_def,
                    [n.name[:-2] for n in input_tensors],
                    [n.name[:-2] for n in output_tensors],
                    [dtype.as_datatype_enum for dtype in dtypes],
                    *toco_args)

            with gfile.FastGFile(filename, "wb") as f:
                f.write(frozen_graph_def.SerializeToString())
                logger.info("Output graph written to {}.".format(filename))

    def export_serving(self, filename,
                       tags=(tf.saved_model.SERVING if is_tfv2() else tf.saved_model.tag_constants.SERVING,),
                       signature_name='prediction_pipeline'):
        """
        Converts a checkpoint and graph to a servable for TensorFlow Serving.
        Use TF's `SavedModelBuilder` to export a trained model without tensorpack dependency.

        Args:
            filename (str): path for export directory
            tags (tuple): tuple of user specified tags
            signature_name (str): name of signature for prediction

        Note:
            This produces

            .. code-block:: none

                variables/       # output from the vanilla Saver
                    variables.data-?????-of-?????
                    variables.index
                saved_model.pb   # a `SavedModel` protobuf

            Currently, we only support a single signature, which is the general PredictSignatureDef:
            https://github.com/tensorflow/serving/blob/master/tensorflow_serving/g3doc/signature_defs.md
        """

        self.graph = self.config._maybe_create_graph()
        with self.graph.as_default():
            input = PlaceholderInput()
            input.setup(self.config.input_signature)
            with PredictTowerContext(''):
                self.config.tower_func(*input.get_input_tensors())

            input_tensors = get_tensors_by_names(self.config.input_names)
            saved_model = tfv1.saved_model.utils
            inputs_signatures = {t.name: saved_model.build_tensor_info(t) for t in input_tensors}
            output_tensors = get_tensors_by_names(self.config.output_names)
            outputs_signatures = {t.name: saved_model.build_tensor_info(t) for t in output_tensors}

            self.config.session_init._setup_graph()
            # we cannot use "self.config.session_creator.create_session()" here since it finalizes the graph
            sess = tfv1.Session(config=tfv1.ConfigProto(allow_soft_placement=True))
            self.config.session_init._run_init(sess)

            builder = tfv1.saved_model.builder.SavedModelBuilder(filename)

            prediction_signature = tfv1.saved_model.signature_def_utils.build_signature_def(
                inputs=inputs_signatures,
                outputs=outputs_signatures,
                method_name=tfv1.saved_model.signature_constants.PREDICT_METHOD_NAME)

            builder.add_meta_graph_and_variables(
                sess, list(tags),
                signature_def_map={signature_name: prediction_signature})
            builder.save()
            logger.info("SavedModel created at {}.".format(filename))
