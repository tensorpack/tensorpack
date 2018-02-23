
# Save and Load models

## Work with TF Checkpoint

The `ModelSaver` callback saves the model to `logger.get_logger_dir()`,
in TensorFlow checkpoint format.
One checkpoint typically includes a `.data-xxxxx` file and a `.index` file.
Both are necessary.

`tf.train.NewCheckpointReader` is the best tool to parse TensorFlow checkpoint.
We have two example scripts to demo its usage, but read [TF docs](https://www.tensorflow.org/api_docs/python/tf/train/NewCheckpointReader) for details.

[scripts/ls-checkpoint.py](../scripts/ls-checkpoint.py)
demos how to print all variables and their shapes in a checkpoint.

[scripts/dump-model-params.py](../scripts/dump-model-params.py) can be used to remove unnecessary variables in a checkpoint.
It takes a metagraph file (which is also saved by `ModelSaver`) and only saves variables that the model needs at inference time.
It can dump the model to a `var-name: value` dict saved in npz format.

## Load a Model

Model loading (in either training or testing) is through the `session_init` interface.
Currently there are two ways a session can be restored:
[session_init=SaverRestore(...)](../modules/tfutils.html#tensorpack.tfutils.sessinit.SaverRestore)
which restores a TF checkpoint,
or [session_init=DictRestore(...)](../modules/tfutils.html#tensorpack.tfutils.sessinit.DictRestore) which restores a dict
([get_model_loader](../modules/tfutils.html#tensorpack.tfutils.sessinit.get_model_loader)
is a small helper to decide which one to use from a file name).
To load multiple models, use [ChainInit](../modules/tfutils.html#tensorpack.tfutils.sessinit.ChainInit).


Variable restoring is completely based on __name match__ between
variables in the current graph and variables in the `session_init` initializer.
Variables that appear in only one side will be printed as warning.

## Transfer Learning
Therefore, transfer learning is trivial.
If you want to load some model, just use the same variable names.
If you want to re-train some layer, just rename it.
