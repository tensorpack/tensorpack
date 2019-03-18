
# Save and Load models

## Inspect a TF Checkpoint

The `ModelSaver` callback saves the model to the directory defined by `logger.get_logger_dir()`,
in TensorFlow checkpoint format.
A TF checkpoint typically includes a `.data-xxxxx` file and a `.index` file.
Both are necessary.

`tf.train.NewCheckpointReader` is the offical tool to parse TensorFlow checkpoint.
Read [TF docs](https://www.tensorflow.org/api_docs/python/tf/train/NewCheckpointReader) for details.
Tensorpack also provides a small tool to load checkpoints, see
[load_chkpt_vars](../modules/tfutils.html#tensorpack.tfutils.varmanip.load_chkpt_vars)
for details.

[scripts/ls-checkpoint.py](../scripts/ls-checkpoint.py)
demos how to print all variables and their shapes in a checkpoint.

[scripts/dump-model-params.py](../scripts/dump-model-params.py) can be used to remove unnecessary variables in a checkpoint.
It takes a metagraph file (which is also saved by `ModelSaver`) and only saves variables that the model needs at inference time.
It dumps the model to a `var-name: value` dict saved in npz format.

## Load a Model to a Session

Model loading (in both training and inference) is through the `session_init` interface.
Currently there are two ways a session can be restored:
[session_init=SaverRestore(...)](../modules/tfutils.html#tensorpack.tfutils.sessinit.SaverRestore)
which restores a TF checkpoint,
or [session_init=DictRestore(...)](../modules/tfutils.html#tensorpack.tfutils.sessinit.DictRestore) which restores a dict.
To load multiple models, use [ChainInit](../modules/tfutils.html#tensorpack.tfutils.sessinit.ChainInit).

Many models in tensorpack model zoo are provided in the form of numpy dictionary (`.npz`),
because it is easier to load and manipulate without requiring TensorFlow.
To load such files to a session, use `DictRestore(dict(np.load(filename)))`.
You can also use
[get_model_loader](../modules/tfutils.html#tensorpack.tfutils.sessinit.get_model_loader),
a small helper to create a `SaverRestore` or `DictRestore` based on the file name.

`DictRestore` is the most general loader because you can make arbitrary changes
you need (e.g., remove variables, rename variables) to the dict.
To load a TF checkpoint into a dict in order to make changes, use
[load_chkpt_vars](../modules/tfutils.html#tensorpack.tfutils.varmanip.load_chkpt_vars).

Variable restoring is completely based on __name match__ between
variables in the current graph and variables in the `session_init` initializer.
Variables that appear in only one side will be printed as warning.

## Transfer Learning
Therefore, transfer learning is trivial.
If you want to load a pre-trained model, just use the same variable names.
If you want to re-train some layer, just rename either the variables in the
graph or the variables in your loader.


## Resume Training

"resume training" is mostly just "loading the last known checkpoint".
Therefore you should refer to the [previous section](#load-a-model-to-a-session)
on how to load a model.

```eval_rst
.. note:: **A checkpoint does not resume everything!**

    The TensorFlow checkpoint only saves TensorFlow variables,
    which means other Python state that are not TensorFlow variables will not be saved
    and resumed. This means:

    1. Training epoch number will not be resumed.
       You can set it by providing a ``starting_epoch`` to your resume job.
    2. State in your callbacks will not be resumed. Certain callbacks maintain a state
       (e.g., current best accuracy) in Python, which cannot be saved automatically.
```


The [AutoResumeTrainConfig](../modules/train.html#tensorpack.train.AutoResumeTrainConfig)
is an alternative of `TrainConfig` which applies some heuristics to
