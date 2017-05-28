
# Input Pipeline

This tutorial covers some general basics of the possible methods to send data from external sources to TensorFlow graph,
and how tensorpack support these methods.
You don't have to read it because these are details under the tensorpack interface,
but knowing it could help understand the efficiency and choose the best input pipeline for your task.

## Prepare Data in Parallel

<!--
   -![prefetch](input-source.png)
	 -->

![prefetch](https://cloud.githubusercontent.com/assets/1381301/26525192/36e5de48-4304-11e7-88ab-3b790bd0e028.png)

A common sense no matter what framework you use:
Start to prepare the next (batch of) data while you're training!

The reasons are:
1. Data preparation often consumes non-trivial time (depend on the actual problem).
2. Data preparation often uses completely different resources from training --
	doing them together doesn't slow you down. In fact you can further parallelize different stages in
	the preparation, because they also use different resources (as shown in the figure).
3. Data preparation often doesn't depend on the result of the previous training step.

Let's do some simple math: according to [tensorflow/benchmarks](https://www.tensorflow.org/performance/benchmarks),
4 P100 GPUs can train ResNet50 at 852 images/sec, and the size of those images are 852\*224\*224\*3\*4bytes = 489MB.
Assuming you have 5GB/s `memcpy` bandwidth, simply copying the data once would take 0.1s -- slowing
down your training by 10%. Think about how many more copies are made during your preprocessing.

Failure to hide the data preparation latency is the major reason why people
cannot see good GPU utilization. Always choose a framework that allows latency hiding.

## Python or C++ ?

The above discussion is valid regardless of what you use to load/preprocess, Python code or TensorFlow operators (written in C++).

The benefit of using TensorFlow ops is:
* Faster preprocessing.
* No "Copy to TF" (i.e. `feed_dict`) stage.

While Python is much easier to write, and has much more libraries to use.

Though C++ ops are potentially faster, they're usually __not necessary__.
As long as data preparation runs faster than training, it makes no difference at all.
And for most types of problems, up to the scale of multi-GPU ImageNet training,
Python can offer enough speed if written properly (e.g. use `tensorpack.dataflow`).
See the [Efficient DataFlow](http://tensorpack.readthedocs.io/en/latest/tutorial/efficient-dataflow.html) tutorial.

When you use Python to load/preprocess data, TF `QueueBase` can help hide the "Copy to TF" latency,
and TF `StagingArea` can help hide the "Copy to GPU" latency.
They are used by most examples in tensorpack,
however most other TensorFlow wrappers are `feed_dict` based -- no latency hiding at all.
This is the major reason why tensorpack is [faster](https://gist.github.com/ppwwyyxx/8d95da79f8d97036a7d67c2416c851b6).

## InputSource

`InputSource` is an abstract interface in tensorpack, to describe where the input come from and how they enter the graph.
For example,

1. Come from a DataFlow and been fed to the graph.
2. Come from a DataFlow and been prefetched on CPU by a TF queue.
3. Come from a DataFlow, prefetched on CPU by a TF queue, then prefetched on GPU by a TF StagingArea.
4. Come from some TF native reading pipeline.
5. Come from some ZMQ pipe, where the load/preprocessing may happen on a different machine.

You can use `TrainConfig(data=)` option to use a customized `InputSource`.
Usually you don't need this API, and only have to specify `TrainConfig(dataflow=)`, because
tensorpack trainers automatically adds proper prefetching for you.
In cases you want to use TF ops rather than DataFlow, you can use `TensorInput` as the `InputSource`
(See the [PTB example](https://github.com/ppwwyyxx/tensorpack/tree/master/examples/PennTreebank)).

## Figure out the Bottleneck

Training and data preparation run in parallel and the faster one will block to wait for the slower one.
So the overall throughput will be dominated by the slower one.

There is no way to accurately benchmark two threads waiting on queues,
without introducing overhead. However, there are ways to understand which one is the bottleneck:

1. Use the average occupancy (size) of the queue. This information is summarized in tensorpack by default.
	If the queue is nearly empty (default size is 50), then the input source is the bottleneck.

2. Benchmark them separately. Use `TestDataSpeed` to benchmark a DataFlow.
	 Use `FakeData(..., random=False)` as a fast DataFlow, to benchmark the training iterations plus the copies.
	 Or use `DummyConstantInput` as a fast InputSource, to benchmark the training iterations only.
