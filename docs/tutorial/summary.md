# Summary and Logging

During training, everything other than the training iterations themselves are executed through callbacks, including all the logging.
This tutorial will explain how summaries and logging are handled in callbacks and how can you customize them.
The default logging behavior should be good enough for normal use cases, so you may skip this tutorial.

### TensorFlow Summaries

This is how TensorFlow summaries eventually get logged/saved/printed:

1. __What to Log__: When you call `tf.summary.xxx` in your graph code, TensorFlow adds an op to
	`tf.GraphKeys.SUMMARIES` collection (by default).
2. __When to Log__: A [MergeAllSummaries](../modules/callbacks.html#tensorpack.callbacks.MergeAllSummaries)
	callback is enabled by default in `TrainConfig`.
	It runs ops in the `SUMMARIES` collection (by default) every epoch (by default),
	and writes results to the monitors.
3. __Where to Log__:
	Several monitor instances are enabled by default in [TrainConfig](../modules/train.html#tensorpack.train.TrainConfig):
	* A [TFEventWriter](../modules/callbacks.html#tensorpack.callbacks.TFEventWriter)
		writes things to an event file used by tensorboard.
	* A [ScalarPrinter](../modules/callbacks.html#tensorpack.callbacks.ScalarPrinter)
		prints all scalars in your terminal.
	* A [JSONWriter](../modules/callbacks.html#tensorpack.callbacks.JSONWriter)
		saves scalars to a JSON file.

All the "what, when, where" can be customized in either the graph or the `TrainConfig`.

Since TF summaries are evaluated every epoch by default, if the content is data-dependent, the results
are likely to have too much variance. To address this issue, you can:
1. Change "When to Log": log more frequently, but note that certain summaries can be expensive to
	 log. You may want to use a separate collection for frequent logging.
2. Change "What to Log": you can call
	 [tfutils.summary.add_moving_summary](../modules/tfutils.html#tensorpack.tfutils.summary.add_moving_summary)
	 on scalar tensors, which will summarize the moving average of those scalars, instead of their instant values.
	 The moving averages are maintained by the
	 [MovingAverageSummary](../modules/callbacks.html#tensorpack.callbacks.MovingAverageSummary)
	 callback (enabled by default).

### Other Data

Besides TensorFlow summaries,
a callback can also write other data to the monitor backend anytime after the training has started.
As long as the type of data is supported, the data will be dispatched to and logged to the same place.

As a result, tensorboard will show not only summaries in the graph, but also your custom data.
For example, a precise validation error often needs to be computed manually, outside the TensorFlow graph.
With a uniform monitor backend, this number will show up in tensorboard as well.
