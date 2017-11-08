
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

## How to print/dump intermediate results in training

1. Learn `tf.Print`.

2. Know [DumpTensors](http://tensorpack.readthedocs.io/en/latest/modules/callbacks.html#tensorpack.callbacks.DumpTensors[]),
	[ProcessTensors](http://tensorpack.readthedocs.io/en/latest/modules/callbacks.html#tensorpack.callbacks.ProcessTensors) callbacks.
	And it's also easy to write your own version of them.

3. The [ProgressBar](http://tensorpack.readthedocs.io/en/latest/modules/callbacks.html#tensorpack.callbacks.ProgressBar)
	 callback can print some scalar statistics, though not enabled by default.

## How to freeze some variables in training

1. You can simply use `tf.stop_gradient` in your model code in some situations (e.g. to freeze first several layers).

2. [varreplace.freeze_variables](../modules/tfutils.html#tensorpack.tfutils.varreplace.freeze_variables) can wrap some variables with `tf.stop_gradient`.

3. [ScaleGradient](../modules/tfutils.html#tensorpack.tfutils.gradproc.ScaleGradient) can be used to set the gradients of some variables to 0.

Note that the above methods only prevent variables being updated by SGD.
Some variables may be updated by other means,
e.g., BatchNorm statistics are updated through the `UPDATE_OPS` collection and the [RunUpdateOps](../modules/callbacks.html#tensorpack.callbacks.RunUpdateOps) callback.

## My training is slow!

Checkout the [Performance Tuning tutorial](performance-tuning.html)
