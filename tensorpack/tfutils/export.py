# -*- coding: UTF-8 -*-
# File: export.py
# Author: Patrick Wieschollek <mail@patwie.com>

"""
This simplifies the process of exporting a model for TensorFlow serving.

"""

import tensorflow as tf
from ..utils import logger
from ..graph_builder.model_desc import ModelDescBase
from ..graph_builder.input_source import PlaceholderInput
from ..tfutils import TowerContext, sessinit


__all__ = ['ModelExport']


class ModelExport(object):
    """Wrapper for tf.saved_model"""
    def __init__(self, model, input_names, output_names):
        """Initialise the export process.

        Example:

            .. code-block:: python
                from mnist_superresolution import Model
                from tensorpack.tfutils import export

                e = ModelExport(Model(), ['lowres'], ['prediction'])
                e.export('train_log/mnist_superresolution/checkpoint', 'export/first_export')

            Will generate a model for TensorFlow serving with input 'lowres' and
            output 'prediction'. The model is in the directory 'export' and can be
            loaded by

            .. code-block:: python

                import tensorflow as tf
                from tensorflow.python.saved_model import tag_constants

                export_dir = 'export/first_export'
                with tf.Session(graph=tf.Graph(), config=tf.ConfigProto(allow_soft_placement=True)) as sess:
                    tf.saved_model.loader.load(sess, [tag_constants.SERVING], export_dir)

                    prediction = tf.get_default_graph().get_tensor_by_name('prediction:0')
                    lowres = tf.get_default_graph().get_tensor_by_name('lowres:0')

                    prediction = sess.run(prediction, {lowres: ...})[0]

        Args:
            model (ModelDescBase): the model description which should be exported
            input_names (list(str)): names of input tensors
            output_names (list(str)): names of output tensors
        """

        assert isinstance(input_names, list)
        assert isinstance(output_names, list)
        assert isinstance(model, ModelDescBase)

        logger.info('[export] prepare new model export')
        super(ModelExport, self).__init__()
        self.model = model
        self.input = PlaceholderInput()
        self.input.setup(self.model.get_inputs_desc())
        self.output_names = output_names
        self.input_names = input_names

    def export(self, checkpoint, export_path, version=1, tags=[tf.saved_model.tag_constants.SERVING],
               signature_name='prediction_pipeline'):
        """Use SavedModelBuilder to export a trained model without TensorPack depency.

        Remarks:
            This produces
                variables/       # output from the vanilla Saver
                    variables.data-?????-of-?????
                    variables.index
                saved_model.pb   # saved model in protcol buffer format

            Currently, we only support a single signature, which is the general PredictSignatureDef:
            https://github.com/tensorflow/serving/blob/master/tensorflow_serving/g3doc/signature_defs.md

        Args:
            checkpoint (str): path to checkpoint file
            export_path (str): path for export directory
            tags (list): list of user specified tags
            signature_name (str): name of signature for prediction
        """
        logger.info('[export] build model for %s' % checkpoint)
        with TowerContext('', is_training=False):
            self.model.build_graph(self.input)

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

            logger.info('[export] exporting trained model to %s' % export_path)
            builder = tf.saved_model.builder.SavedModelBuilder(export_path)

            logger.info('[export] build signatures')
            # build inputs
            inputs_signature = dict()
            for n, v in zip(self.input_names, self.inputs):
                logger.info('[export] add input signature: %s' % v)
                inputs_signature[n] = tf.saved_model.utils.build_tensor_info(v)

            outputs_signature = dict()
            for n, v in zip(self.output_names, self.outputs):
                logger.info('[export] add output signature: %s' % v)
                outputs_signature[n] = tf.saved_model.utils.build_tensor_info(v)

            prediction_signature = tf.saved_model.signature_def_utils.build_signature_def(
                inputs=inputs_signature,
                outputs=outputs_signature,
                method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)

            # legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')

            builder.add_meta_graph_and_variables(
                self.sess, tags,
                signature_def_map={signature_name: prediction_signature})
            builder.save()
