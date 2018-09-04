# Export Models

You might want to export your trained model with its graph to use it in production.
This tutorial gives an overview of how to export your model trained in Tensorpack. Tensorpack offers an unique advantage as it is designed to rely on a `ModelDescr` class representing your model and a `PredictConfig` gathering all necessary information to build the inference only-graph.

Currently, there are the following ways to save your model:

- Use the `ModelSaver` callback to save checkpoints during training.
- Use the `???script` to save all weights without the graph into a `npz` file similar to amost pre-trained models offered by Tensorpack.
- Use the `???export_script` to generate a `tf.SavedModel` which offers a plug-and-play solution with TensorFlow-Serving.
- Use the `???export_script` to generate a freezed model which is usefule when working on mobile apps.



# Saving Checkpoints (ModelSaver)

During training Tensorpack can already use the `ModelSaver`-callback which regulary saves snapshots of your model onto disk.
This topic is already covered in [Save and Load models](save-load.html).

# Exporting the weights into npz files

Pre-trained models in Tensorpack are packed as npz, such that you can re-use the weights without TensorFlow dependencies.
