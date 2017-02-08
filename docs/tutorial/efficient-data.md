
# Efficient Data Loading

This tutorial gives an overview of how to efficiently load data in tensorpack, using ImageNet
dataset as an example.

Note that the actual performance would depend on not only the disk, but also
memory (for caching) and CPU (for data processing), so the solution in this tutorial is
not necessarily the best for different scenarios.

### Use TensorFlow queues

In general, ``feed_dict`` is slow and should never appear in your critical loop.
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

This is automatically handled by tensorpack trainers already (unless you used the demo ``SimpleTrainer``),
see [Trainer](trainer.md) for details.
TensorFlow is providing staging interface which may further improve the speed. This is
[issue#140](https://github.com/ppwwyyxx/tensorpack/issues/140).

### Figure out your bottleneck

