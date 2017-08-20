
# Input Pipeline

This tutorial contains some general discussions on the topic of
"how to read data efficiently to work with TensorFlow",
and how tensorpack support these methods.
You don't have to read it because these are details under the tensorpack interface,
but knowing it could help understand the efficiency and choose the best input pipeline for your task.

## Prepare Data in Parallel

![prefetch](https://cloud.githubusercontent.com/assets/1381301/26525192/36e5de48-4304-11e7-88ab-3b790bd0e028.png)

A common sense no matter what framework you use:
<center>
Prepare data in parallel with the training!
</center>

The reasons are:
1. Data preparation often consumes non-trivial time (depend on the actual problem).
2. Data preparation often uses completely different resources from training (see figure above) --
	doing them together doesn't slow you down. In fact you can further parallelize different stages in
	the preparation since they also use different resources.
3. Data preparation often doesn't depend on the result of the previous training step.

Let's do some simple math: according to [tensorflow/benchmarks](https://www.tensorflow.org/performance/benchmarks),
4 P100 GPUs can train ResNet50 at 852 images/sec, and the size of those images are 852\*224\*224\*3\*4bytes = 489MB.
Assuming you have 5GB/s `memcpy` bandwidth, simply copying the data once would take 0.1s -- slowing
down your training by 10%. Think about how many more copies are made during your preprocessing.

Failure to hide the data preparation latency is the major reason why people
cannot see good GPU utilization. __Always choose a framework that allows latency hiding.__
However most other TensorFlow wrappers are designed to be `feed_dict` based.
This is the major reason why tensorpack is [faster](https://gist.github.com/ppwwyyxx/8d95da79f8d97036a7d67c2416c851b6).

## Python Reader or TF Reader ?

The above discussion is valid regardless of what you use to load/preprocess data,
either Python code or TensorFlow operators (written in C++).

The benefits of using TensorFlow ops are:
* Faster read/preprocessing.

	* Potentially true, but not necessarily. With Python code you can call a variety of other fast libraries, which
		you have no access to in TF ops. For example, LMDB could be faster than TFRecords.
	* Python may be just fast enough.

		As long as data preparation runs faster than training, and the latency of all four blocks in the
		above figure is hidden, it makes no difference at all.
		For most types of problems, up to the scale of multi-GPU ImageNet training,
		Python can offer enough speed if you use a fast library (e.g. `tensorpack.dataflow`).
		See the [Efficient DataFlow](efficient-dataflow.html) tutorial
		on how to build a fast Python reader with DataFlow.

* No "Copy to TF" (i.e. `feed_dict`) stage.

	* True. But as mentioned above, the latency can usually be hidden.

		In tensorpack, TF queues are used to hide the "Copy to TF" latency,
		and TF `StagingArea` can help hide the "Copy to GPU" latency.
		They are used by most examples in tensorpack.

The benefits of using Python reader is obvious:
it's much much easier to write Python to read different data format,
handle corner cases in noisy data, preprocess, etc.

## InputSource

`InputSource` is an abstract interface in tensorpack, to describe where the input come from and how they enter the graph.
For example,

1. Come from a DataFlow and been fed to the graph.
2. Come from a DataFlow and been prefetched on CPU by a TF queue.
3. Come from a DataFlow, prefetched on CPU by a TF queue, then prefetched on GPU by a TF StagingArea.
4. Come from some TF native reading pipeline.
5. Come from some ZMQ pipe, where the load/preprocessing may happen on a different machine.

When you set `TrainConfig(dataflow=)`, tensorpack trainers automatically adds proper prefetching for you.
You can also use `TrainConfig(data=)` option to use a customized `InputSource`.
In case you want to use TF ops rather than a DataFlow, you can use `TensorInput` as the `InputSource`
(See the [PTB example](../../tensorpack/tree/master/examples/PennTreebank)).

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
