# Summary and Logging

During training, everything other than the iterations are executed through callbacks.
This tutorial will explain how summaries and logging are handled in callbacks and how can you customize them.
The default logging behavior should be good enough for normal use cases, so you may skip this tutorial.

### TensorFlow Summaries

This is how TensorFlow summaries eventually get logged/saved/printed:

1. __What to Log__: Define what you want to log in the graph. 
   When you call `tf.summary.xxx` in your graph code, TensorFlow adds an op to
	`tf.GraphKeys.SUMMARIES` collection (by default).
2. __When to Log__: [MergeAllSummaries](../modules/callbacks.html#tensorpack.callbacks.MergeAllSummaries)
	callback is one of the [default callbacks](../modules/train.html#tensorpack.train.DEFAULT_CALLBACKS).
	It runs ops in the `tf.GraphKeys.SUMMARIES` collection (by default) every epoch (by default),
	and writes results to the monitors.
3. __Where to Log__:
	Several monitors are [enabled by default](../modules/train.html#tensorpack.train.DEFAULT_MONITORS).
	* A [TFEventWriter](../modules/callbacks.html#tensorpack.callbacks.TFEventWriter)
		writes things to an event file used by tensorboard.
	* A [ScalarPrinter](../modules/callbacks.html#tensorpack.callbacks.ScalarPrinter)
		prints all scalars in your terminal.
	* A [JSONWriter](../modules/callbacks.html#tensorpack.callbacks.JSONWriter)
		saves scalars to a JSON file.

All the "what, when, where" can be customized in either the graph or with the callbacks/monitors setting.

The design goal to disentangle "what, when, where" is to make components reusable.
Suppose you have `M` items to log 
(possibly from differently places, not necessarily the graph)
and `N` backends to log your data to, you
automatically obtain all the `MxN` combinations.

Despite of that, if you only care about logging one specific item (e.g. for
debugging purpose), you can check out the 
[FAQ](http://tensorpack.readthedocs.io/tutorial/faq.html#how-to-print-dump-intermediate-results-in-training)
for easier options.

### Noisy TensorFlow Summaries

Since TF summaries are evaluated infrequently (every epoch) by default,
if the content is data-dependent, the values could have high variance.
To address this issue, you can:
1. Change "When to Log": log more frequently, but note that certain summaries can be expensive to
  log. You may want to use a separate collection for frequent logging.
2. Change "What to Log": you can call
  [tfutils.summary.add_moving_summary](../modules/tfutils.html#tensorpack.tfutils.summary.add_moving_summary)
  on scalar tensors, which will summarize the moving average of those scalars, instead of their instant values.
  The moving averages are maintained by the
  [MovingAverageSummary](../modules/callbacks.html#tensorpack.callbacks.MovingAverageSummary)
  callback (enabled by default).

### Other Logging Data

Besides TensorFlow summaries,
a callback can also write other data to the monitor backend anytime once the training has started,
by `self.trainer.monitors.put_xxx`.
As long as the type of data is supported, the data will be dispatched to and logged to the same place.

As a result, tensorboard will show not only summaries in the graph, but also your custom data.
For example, a precise validation error often needs to be computed manually, outside the TensorFlow graph.
With a uniform monitor backend, this number will show up in tensorboard as well.

### Textual Logs

```python
from tensorpack.utils import logger
# logger has the methods of Python's logging.Logger
logger.info("Hello World!")
```

See [APIs of utils.logger](../modules/utils.html#module-tensorpack.utils.logger)
