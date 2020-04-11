
# Save and Load models

## Work with a TF Checkpoint

The `ModelSaver` callback saves the model to the directory defined by `logger.get_logger_dir()`,
in TensorFlow checkpoint format.
A TF checkpoint typically includes a `.data-xxxxx` file and a `.index` file.
Both are necessary.

`tf.train.NewCheckpointReader` is the offical tool to parse TensorFlow checkpoint.
Read [TF docs](https://www.tensorflow.org/api_docs/python/tf/train/NewCheckpointReader) for details.
Tensorpack also provides a small tool to load checkpoints, see
[load_chkpt_vars](../modules/tfutils.html#tensorpack.tfutils.varmanip.load_chkpt_vars)
for details.

[scripts/ls-checkpoint.py](../../scripts/ls-checkpoint.py)
demos how to print all variables and their shapes in a checkpoint.

Tensorpack includes another tool to save variables to TF checkpoint, see
[save_chkpt_vars](../modules/tfutils.html#tensorpack.tfutils.varmanip.save_chkpt_vars).

## Work with .npz Files in the Model Zoo

Most models provided by tensorpack are in npz (dictionary) format,
because it's easy to use without TF dependency.
You can read/write them with `np.load` and `np.savez`.

[scripts/dump-model-params.py](../../scripts/dump-model-params.py) can be used to remove unnecessary variables in a checkpoint
and save results to a npz.
It takes a metagraph file (which is also saved by `ModelSaver`) and only saves variables that the model needs at inference time.
It dumps the model to a `var-name: value` dict saved in npz format.

## Load a Model to a Session

Model loading (in both training and inference) is through the `session_init` interface.
For training, use `session_init` in `TrainConfig(...)` or `Trainer.train(...)`.
For inference, use `session_init` in `PredictConfig(...)`.

There are a few ways a session can be initialized:
```
session_init=SmartInit("path/to/checkpoint")  # load a TF checkpoint
session_init=SmartInit("path/to/model_zoo.npz")  # load tensorpack model zoo
session_init=SmartInit(dict_of_parameters)  # load a dictionary
session_init=SmartInit(["path1", dict2])  # load them sequentially
```

[SmartInit](../modules/tfutils.html#tensorpack.tfutils.sessinit.SmartInit)
is in fact a small helper which uses some heuristics to return you one of
[SaverRestore](../modules/tfutils.html#tensorpack.tfutils.sessinit.SaverRestore) or
[DictRestore](../modules/tfutils.html#tensorpack.tfutils.sessinit.DictRestore).
They are responsible for the actual initialization work.

Whatever you use in `session_init`, this is what happens during the initialization:

* Variable restoring is completely based on __exact name match__ between
	variables in the current graph and variables in the `session_init` initializer.
* Variables that appear in only one side will be printed as warning.
* Variables of the same name but incompatible shapes will cause exceptions.
  If you set `ignore_mismatch=True`, then such errors will only be printed as warnings.

You can also use `SmartInit` to load a model to a session manually,
without involving the rest of the tensorpack, by `SmartInit(...).init(session)`.

## Transfer Learning

Therefore, transfer learning is trivial.

If you want to load a pre-trained model, just use the same variable names.
If you want to re-train some layer, either rename the variables in the
graph, or rename/remove the variables in your loader.


## Resume Training

"Resume training" is mostly just "loading the last known checkpoint".
To load a model, you should refer to the previous section: [Load a Model to a Session](#load-a-model-to-a-session).

```eval_rst
.. note:: **A checkpoint does not resume everything!**

    Loading the checkpoint does most of the work in "resume trainig", but note that
    TensorFlow checkpoint only saves TensorFlow variables,
    which means other Python state that are not TensorFlow variables will not be saved
    and resumed. This means:

    1. Training epoch number will not be resumed.
       You can set it by providing a ``starting_epoch`` to your ``TrainConfig``.
    2. State in your callbacks will not be resumed. Certain callbacks maintain a state
       (e.g., current best accuracy) in Python, which cannot be saved automatically.
```

The [AutoResumeTrainConfig](../modules/train.html#tensorpack.train.AutoResumeTrainConfig)
is an alternative of `TrainConfig` which applies some heuristics to load the lastest epoch number and lastest checkpoint.
