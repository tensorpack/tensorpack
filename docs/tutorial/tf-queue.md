
# How data goes into graph

This tutorial covers how data goes from DataFlow to TensorFlow graph.
They are tensorpack internal details, but it's important to know
if you care about efficiency.

## Use TensorFlow queues

In general, `feed_dict` is slow and should never appear in your critical loop.
i.e., you should avoid loops like this:
```python
while True:
  X, y = get_some_data()
  minimize_op.run(feed_dict={'X': X, 'y': y})
```
However, when you need to load data from Python-side, this is the only available interface in frameworks such as Keras, tflearn.

You should use something like this instead:
```python
# Thread 1:
while True:
  X, y = get_some_data()
  enqueue.run(feed_dict={'X': X, 'y': y})	 # feed data to a TensorFlow queue

# Thread 2:
while True:
  minimize_op.run()	 # minimize_op was built from dequeued tensors
```

This is now automatically handled by tensorpack trainers already (unless you used the demo ``SimpleTrainer``),
see [Trainer](trainer.md) for details.

TensorFlow provides staging interface which will further improve the speed in the future. This is
[issue#140](https://github.com/ppwwyyxx/tensorpack/issues/140).

You can also avoid `feed_dict` by using TensorFlow native operators to read data, which is also
supported here.
It probably allows you to reach the best performance, but at the cost of implementing the
reading / preprocessing ops in C++ if there isn't one for your task. We won't talk about it here.

## Figure out the bottleneck

For training we will only worry about the throughput but not the latency.
Thread 1 & 2 runs in parallel, and the faster one will block to wait for the slower one.
So the overall throughput will appear to be the slower one.

There isn't a way to accurately benchmark the two threads while they are running, without introducing overhead. But
there are ways to understand which one is the bottleneck:

1. Use the average occupancy (size) of the queue. This information is summarized after every epoch (TODO depend on #125).
	If the queue is nearly empty, then the data thread is the bottleneck.

2. Benchmark them separately. You can use `TestDataSpeed` to benchmark a DataFlow, and
	 use `FakeData` as a fast replacement in a dry run to benchmark the training
	 iterations.
