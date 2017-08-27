# Summary and Logging

This tutorial will introduce the `Monitor` backend and
explain how tensorpack handles summaries and logging.

### Monitors

In tensorpack, everything besides the training iterations are done in callbacks, including all the logging.

When a callback gets something to log, it will write to the monitor backend through
`trainer.monitors`, by calling `put_{scalar,image,summary,...}`.
The call gets dispatched to multiple `TrainingMonitor` instances.
These monitors are a special type of callback which can process different types of log data,
and can be customized in `TrainConfig`.


### TensorFlow Summaries

Here is how TensorFlow summaries eventually get logged/saved/printed:

1. __What to Log__: When you call `tf.summary.xxx` in your graph code, TensorFlow adds an op to
	`tf.GraphKeys.SUMMARIES` collection (by default).
2. __When to Log__: A [MergeAllSummaries](../modules/callbacks.html#tensorpack.callbacks.MergeAllSummaries)
	callback is enabled by default in `TrainConfig`.
	It runs ops in the `SUMMARIES` collection (by default) every epoch (by default),
	and writes results to the monitor backend.
3. __Where to Log__:
	* A [TFEventWriter](../modules/callbacks.html#tensorpack.callbacks.TFEventWriter)
		monitor is enabled by default in [TrainConfig](../modules/train.html#tensorpack.train.TrainConfig),
		which writes things to an event file used by tensorboard.
	* A [ScalarPrinter](../modules/callbacks.html#tensorpack.callbacks.ScalarPrinter)
		monitor is enabled by default, which prints all scalars in your terminal.
	* A [JSONWriter](../modules/callbacks.html#tensorpack.callbacks.JSONWriter)
		monitor is enabled by default, which saves scalars to a file.

Since summaries are evaluated every epoch by default, if the content is data-dependent, the results
are likely to have too much variance. You can:
1. Change "When to Log": log more frequently, but note that some large summaries are expensive to
	 log. You may want to use a separate collection for frequent logging.
2. Change "What to Log": you can call
	 [tfutils.summary.add_moving_summary](../modules/tfutils.html#tensorpack.tfutils.summary.add_moving_summary)
	 on scalar tensors, which will summarize the moving average of those scalars instead of their instant values.
	 The moving averages are maintained by the
	 [MovingAverageSummary](../modules/callbacks.html#tensorpack.callbacks.MovingAverageSummary)
	 callback (enabled by default).

Besides TensorFlow summaries,
a callback is free to log any other types of data to the monitor backend,
anytime after the training has started.
As long as the type of data is supported, it will be logged by each monitor.
In other words, tensorboard can show not only summaries in the graph, but also your custom data.
