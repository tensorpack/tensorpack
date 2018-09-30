
# Save and Load models

## Inspect a TF Checkpoint

The `ModelSaver` callback saves the model to the directory defined by `logger.get_logger_dir()`,
in TensorFlow checkpoint format.
A TF checkpoint typically includes a `.data-xxxxx` file and a `.index` file.
Both are necessary.

`tf.train.NewCheckpointReader` is the offical tool to parse TensorFlow checkpoint.
Read [TF docs](https://www.tensorflow.org/api_docs/python/tf/train/NewCheckpointReader) for details.
Tensorpack also provides some small tools to work with checkpoints, see 
[documentation](../modules/tfutils.html#tensorpack.tfutils.varmanip.load_chkpt_vars)
for details.

[scripts/ls-checkpoint.py](../scripts/ls-checkpoint.py)
demos how to print all variables and their shapes in a checkpoint.

[scripts/dump-model-params.py](../scripts/dump-model-params.py) can be used to remove unnecessary variables in a checkpoint.
It takes a metagraph file (which is also saved by `ModelSaver`) and only saves variables that the model needs at inference time.
It can dump the model to a `var-name: value` dict saved in npz format.

## Load a Model to a Session

Model loading (in either training or inference) is through the `session_init` interface.
Currently there are two ways a session can be restored:
[session_init=SaverRestore(...)](../modules/tfutils.html#tensorpack.tfutils.sessinit.SaverRestore)
which restores a TF checkpoint,
or [session_init=DictRestore(...)](../modules/tfutils.html#tensorpack.tfutils.sessinit.DictRestore) which restores a dict.
[get_model_loader](../modules/tfutils.html#tensorpack.tfutils.sessinit.get_model_loader)
is a small helper to decide which one to use from a file name.
To load multiple models, use [ChainInit](../modules/tfutils.html#tensorpack.tfutils.sessinit.ChainInit).


Variable restoring is completely based on __name match__ between
variables in the current graph and variables in the `session_init` initializer.
Variables that appear in only one side will be printed as warning.

## Transfer Learning
Therefore, transfer learning is trivial.
If you want to load a pre-trained model, just use the same variable names.
If you want to re-train some layer, just rename it.
