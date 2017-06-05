# -*- coding: UTF-8 -*-
# File: exporter.py
# Author: Patrick Wieschollek <mail@patwie.com>

"""
This simplifies the process of exporting a model for TensorFlow serving.

"""

import tensorflow as tf
from tensorpack.utils import logger
from tensorpack.tfutils import TowerContext
from tensorpack.models import ModelDesc
from tensorpack.tfutils import sessinit
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import signature_constants, signature_def_utils, tag_constants, utils


__all__ = ['ModelExport']


class ModelExport(object):
    """Wrapper for tf.saved_model"""
    def __init__(self, model, input_names, output_names):
        """Initialise the export process.

        Example:

            .. code-block:: python
                from mnist_superresolution import Model
                from exporter import ModelExport

                e = ModelExport(Model, ['lowres'], ['prediction'])
                e.build('train_log/mnist_superresolution/checkpoint')
                e.export('export')

            Will generate a model for TensorFlow serving with input 'lowres' and
            output 'prediction'. The model is in the directory 'export' and can be
            loaded by

            .. code-block:: python

                import tensorflow as tf
                from tensorflow.python.saved_model import tag_constants

                export_dir = "export"
                with tf.Session(graph=tf.Graph(), config=tf.ConfigProto(allow_soft_placement=True)) as sess:
                    tf.saved_model.loader.load(sess, [tag_constants.SERVING], export_dir)

                    prediction = tf.get_default_graph().get_tensor_by_name('prediction:0')
                    lowres = tf.get_default_graph().get_tensor_by_name('lowres:0')

                    prediction = sess.run(prediction, {lowres: ...})[0]

        Args:
            model (ModelDescr): the model description which should be exported
            input_names (list(str)): names of input tensors
            output_names (list(str)): names of output tensors
        """

        assert isinstance(input_names, list)
        assert isinstance(output_names, list)

        logger.info('[export] prepare new model export')
        super(ModelExport, self).__init__()
        self.model = model()
        assert isinstance(self.model, ModelDesc)
        self.placehdrs = self.model.get_reused_placehdrs()
        self.output_names = output_names
        self.input_names = input_names

    def build(self, checkpoint):
        """Summary

        Args:
            checkpoint (str): path to checkpoint file
        """
        logger.info('[export] build model for %s' % checkpoint)
        with TowerContext('', is_training=False):
            self.model._build_graph(self.placehdrs)

            self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
            # load values from latest checkpoint
            init = sessinit.SaverRestore(checkpoint)
            self.sess.run(tf.global_variables_initializer())
            init.init(self.sess)

            self.inputs = []
            for n in self.input_names:
                tensor = tf.get_default_graph().get_tensor_by_name('%s:0' % n)
                logger.info('[export] add input-tensor "%s"' % tensor.name)
                self.inputs.append(tensor)

            self.outputs = []
            for n in self.output_names:
                tensor = tf.get_default_graph().get_tensor_by_name('%s:0' % n)
                logger.info('[export] add output-tensor "%s"' % tensor.name)
                self.outputs.append(tensor)

    def export(self, export_path, version=1):
        """Write all files for exported model

        Args:
            export_path (str): path for export directory
        """
        logger.info('[export] exporting trained model to %s' % export_path)
        builder = saved_model_builder.SavedModelBuilder(export_path)

        logger.info('[export] build signatures')
        # build inputs
        inputs_signature = dict()
        for n, v in zip(self.input_names, self.inputs):
            logger.info('[export] add input signature: %s' % v)
            inputs_signature[n] = utils.build_tensor_info(v)

        outputs_signature = dict()
        for n, v in zip(self.output_names, self.outputs):
            logger.info('[export] add output signature: %s' % v)
            outputs_signature[n] = utils.build_tensor_info(v)

        prediction_signature = signature_def_utils.build_signature_def(
            inputs=inputs_signature,
            outputs=outputs_signature,
            method_name=signature_constants.PREDICT_METHOD_NAME)

        # legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')

        builder.add_meta_graph_and_variables(
            self.sess, [tag_constants.SERVING],
            signature_def_map={'prediction_pipeline': prediction_signature})
        builder.save()
