
# FAQs

## Does it support data format X / augmentation Y / layer Z?

The library tries to support everything, but it couldn't really include everything.

For the XYZ you need, you can either implement them, or use any existing code and wrap it
with tensorpack interface. See [Extend Tensorpack](http://tensorpack.readthedocs.io/en/latest/tutorial/index.html#extend-tensorpack)
for more details.

It you think:
1. The framework has limitation so your XYZ cannot be supported, OR
2. Your XYZ is very common, or very well-defined, so it would be nice to include it.

Then it's a good time to open an issue.

## How to dump/inspect a model

When you enable `ModelSaver` as a callback,
trained models will be stored in TensorFlow checkpoint format, which typically includes a
`.data-xxxxx` file and a `.index` file. Both are necessary.

To inspect a checkpoint, the easiest way is `tf.train.NewCheckpointReader`. Please note that it
expects a path without the extension.

You can dump a cleaner version of the model (with only model/trainable variables), with
`scripts/dump-model-params.py`, as a simple `var-name: value` dict saved in npy format.
The script expects a metagraph file which is also saved by `ModelSaver`.


## How to load a model / do transfer learning

All model loading (in either training or testing) is through the `session_init` interface
in `TrainConfig` or `PredictConfig`.
It accepts a `SessionInit` instance, where the common options are `SaverRestore` which restores
TF checkpoint, or `DictRestore` which restores a dict. `get_model_loader` is a small helper to
decide which one to use from file name.

Doing transfer learning is painless. Variable restoring is completely based on name match between
the current graph and the `SessionInit` initializer.
Therefore, if you want to re-train some layer, just rename it.
And unmatched variables on both side will be printed as warning.

To freeze some variables, there are [different ways](https://github.com/ppwwyyxx/tensorpack/issues/87#issuecomment-270545291)
with pros and cons.
