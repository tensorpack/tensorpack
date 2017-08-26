
# FAQs

## Does it support data format X / augmentation Y / layer Z?

The library tries to __support__ everything, but it could not really __include__ everything.

The interface attempts to be flexible enough so you can put any XYZ on it.
You can either implement them under the interface or simply wrap some existing Python code.
See [Extend Tensorpack](index.html#extend-tensorpack)
for more details.

If you think:
1. The framework has limitation in its interface so your XYZ cannot be supported, OR
2. Your XYZ is super common / very well-defined / very useful, so it would be nice to include it.

Then it is a good time to open an issue.

## How to dump/inspect a model

When you enable `ModelSaver` as a callback,
trained models will be stored in TensorFlow checkpoint format, which typically includes a
`.data-xxxxx` file and a `.index` file. Both are necessary.

To inspect a checkpoint, the easiest tool is `tf.train.NewCheckpointReader`. Please note that it
expects a model path without the extension.

You can dump a cleaner version of the model (without unnecessary variables), using
`scripts/dump-model-params.py`, as a simple `var-name: value` dict saved in npy/npz format.
The script expects a metagraph file which is also saved by `ModelSaver`.


## How to load a model / do transfer learning

All model loading (in either training or testing) is through the `session_init` initializer
in `TrainConfig` or `PredictConfig`.
The common choices for this option are `SaverRestore` which restores a
TF checkpoint, or `DictRestore` which restores a dict. (`get_model_loader` is a small helper to
decide which one to use from a file name.)

Doing transfer learning is trivial.
Variable restoring is completely based on name match between
the current graph and the `SessionInit` initializer.
Therefore, if you want to load some model, just use the same variable name
so the old value will be loaded into the variable.
If you want to re-train some layer, just rename it.
Unmatched variables on both sides will be printed as a warning.

## How to freeze some variables in training

1. You can simply use `tf.stop_gradient` in your model code in some situations (e.g. to freeze first several layers).

2. [varreplace.freeze_variables](../modules/tfutils.html#tensorpack.tfutils.varreplace.freeze_variables) can wrap some variables with `tf.stop_gradient`.

3. [ScaleGradient](../modules/tfutils.html#tensorpack.tfutils.gradproc.ScaleGradient) can be used to set the gradients of some variables to 0.

Note that the above methods only prevent variables being updated by SGD.
Some variables may be updated by other means,
e.g., BatchNorm statistics are updated through the `UPDATE_OPS` collection and the [RunUpdateOps](../modules/callbacks.html#tensorpack.callbacks.RunUpdateOps) callback.
