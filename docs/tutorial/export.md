# Export Models

Tensorpack delivers a training framework but also supports the process of preparing a model to be deployed in production either using TensorFlow serving or mobile applications.
Tensorpack offers a unique advantage as it is designed to rely on a `ModelDescr` class representing your model and a `PredictConfig` containing all necessary information to build the inference only-graph.

Currently, there are the following ways to save your model:

- Use the `ModelSaver` callback to save checkpoints during training.
- Use the `dump-model-params.py` script to save all weights without the graph into a compressed numpy container `npz` file similar to most pre-trained models offered by Tensorpack.
- Use the `ModelExporter` to generate a `tf.SavedModel` which offers a plug-and-play solution with TensorFlow-Serving.
- Use the `ModelExporter` which generate a frozen and pruned model ready to deploy in mobile apps.


# Saving Checkpoints (ModelSaver)

During training Tensorpack already offers the `ModelSaver`-callback which regularly saves snapshots of your model onto disk.
This topic is already covered in [Save and Load models](save-load.html).

# Exporting the weights into npz files

Pre-trained models in Tensorpack are packed as `npz` files, such that you can re-use the weights without any TensorFlow dependencies.

Given your checkpoints are stored in the following directory structure

```
train_logs
  some_model
    graph-0904-161912.meta
    checkpoint
    model-1.data-00000-of-00001
    model-1.index
```

run the following command

```console

user@host $ python tensorpack/scripts/dump-model-params.py \
  -- meta train_log/export/graph-0904-161912.meta \
  train_log/export/checkpoint \
  target_file_weights.npz
```

Such a `npz` file can be reused later in Tensorpack.

# Exporting the model for TensorFlow Serving

Exporting the model is similar to apply inference *within* Tensorpack. It requires to write a `PredictConfig` and uses the `ModelExporter` like

```python
pred_config = PredictConfig(
    session_init=get_model_loader(model_path),
    model=YourModel(),
    input_names=['some_input_name'],
    output_names=['some_output_name'])

ModelExporter(pred_config).export_serving('/path/to/export')
```

You might want to rewrite the graph for inference beforehand, e.g., to support base64-encoded images. The example provided in `examples/basic/export.py` demonstrates such an altered inference graph.

# Freezing and pruning a trained model

For mobile and similar applications you might want to change the graph before exporting by:

- Convert all variables to constants to embed the weights directly in the graph.
- Remove all unnecessary operations (training-only ops, e.g., learning-rate) to compress the graph.

This creates a self-contained graph which includes all necessary information to run inference.
Tensorpack's `ModelExporter` takes care of both steps automatically:

```python
pred_config = PredictConfig(
    session_init=get_model_loader(model_path),
    model=YourModel(),
    input_names=['some_input_name'],
    output_names=['some_output_name'])

ModelExporter(pred_config).export_compact('/path/to/compact_graph.pb')
```

Again, `examples/basic/export.py` demonstrates the usage of such a frozen/pruned graph.
