
# Input Sources

This tutorial covers how data goes from DataFlow or other sources to TensorFlow graph.
You don't have to know it, but it may help with efficiency.

`InputSource` is an abstract interface in tensorpack describing where the input come from and how they enter the graph.
For example,

1. Come from a DataFlow and been fed to the graph.
2. Come from a DataFlow and been prefetched on CPU by a TF queue.
3. Come from a DataFlow, prefetched on CPU by a TF queue, then prefetched on GPU by a TF StagingArea.
4. Come from some TF native reading pipeline.
5. Come from some ZMQ pipe.

For most tasks, DataFlow with some prefetch is fast enough. You can use `TrainConfig(data=)` option
to customize your `InputSource`.

## Use Prefetch

In general, `feed_dict` is slow and should never appear in your critical loop.
i.e., when you use TensorFlow without any wrappers, you should avoid loops like this:
```python
while True:
  X, y = get_some_data()
  minimize_op.run(feed_dict={'X': X, 'y': y})
```
However, when you need to load data from Python-side, this is the only available interface in frameworks such as Keras, tflearn.
This is part of the reason why [tensorpack is faster](https://gist.github.com/ppwwyyxx/8d95da79f8d97036a7d67c2416c851b6) than examples from other frameworks.

You should use something like this instead, to prefetch data into the graph in one thread and hide the copy latency:
```python
# Thread 1:
while True:
  X, y = get_some_data()
  enqueue.run(feed_dict={'X': X, 'y': y})	 # feed data to a TensorFlow queue

# Thread 2:
while True:
  minimize_op.run()	 # minimize_op was built from dequeued tensors
```

This is now automatically handled by tensorpack trainers already, see [Trainer](trainer.md) for details.

TensorFlow StagingArea can further hide H2D (CPU->GPU) copy latency.
It is also automatically included in tensorpack when you use Synchronous MultiGPU training.

You can also avoid `feed_dict` by using TensorFlow native operators to read data, which is also supported in tensorpack.
It probably allows you to reach the best performance,
but at the cost of implementing the reading / preprocessing ops in C++ if there isn't one for your task.

## Figure out the bottleneck

Thread 1 & 2 runs in parallel and the faster one will block to wait for the slower one.
So the overall throughput will appear to be the slower one.

There is no way to accurately benchmark the two dependent threads while they are running,
without introducing overhead. However, are ways to understand which one is the bottleneck:

1. Use the average occupancy (size) of the queue. This information is summarized by default.
	If the queue is nearly empty (default size 50), then the input source is the bottleneck.

2. Benchmark them separately. You can use `TestDataSpeed` to benchmark a DataFlow, and
	 use `FakeData` as a fast replacement in a dry run, to benchmark the training iterations.

If you found your input is the bottleneck, then you'll need to think about how to speed up your data.
You may either change `InputSource`, or look at [Efficient DataFlow](http://tensorpack.readthedocs.io/en/latest/tutorial/efficient-dataflow.html).
