
# Performance Tuning

__We do not know why your training is slow__ (and most of the times it's not a tensorpack problem).

Tensorpack is designed to be high-performance, as can be seen in the [benchmarks](https://github.com/tensorpack/benchmarks).
But performance is different across machines and tasks,
so it's not easy to understand what goes wrong without doing some investigations by your own.
Tensorpack has some tools to make it easier to understand the performance.
Here is a list of things you can do to understand why your training is slow.

If you ask for help to understand and improve the speed, PLEASE do the
investigations below, post your hardware information and your findings from the investigation, such as what changes
you've made and what performance numbers you've seen.

## Figure out the bottleneck

1. If you use feed-based input (unrecommended) and datapoints are large, data is likely to become the bottleneck.
2. If you use queue-based input + DataFlow, always pay attention to the queue size statistics in
   training log. Ideally the input queue should be nearly full (default size is 50).
   __If the queue size is close to zero, data is the bottleneck. Otherwise, it's not.__

   The size is by default printed after every epoch. Set `steps_per_epoch` to a
   smaller number (e.g. 100) to see this number earlier.
3. If GPU utilization is low but queue is full, the graph is inefficient.
   Either there are some communication inefficiency, or some ops in the graph are inefficient (e.g. CPU ops). Also make sure GPUs are not locked in P8 state.

## Benchmark the components

Whatever benchmarks you're doing, never look at the speed of the first 50 iterations.
Everything is slow at the beginning.

1. Use `dataflow=FakeData(shapes, random=False)` to replace your original DataFlow by a constant DataFlow.
	This will benchmark the graph, without the possible overhead of DataFlow.
2. (usually not needed) Use `data=DummyConstantInput(shapes)` for training, so that the iterations only take data from a constant tensor.
	No DataFlow is involved in this case.
3. If you're using a TF-based input pipeline you wrote, you can simply run it in a loop and test its speed.
4. Use `TestDataSpeed(mydf).start()` to benchmark your DataFlow.

A benchmark will give you more precise information about which part you should improve.

## Investigate DataFlow

Understand the [Efficient DataFlow](efficient-dataflow.html) tutorial, so you know what your DataFlow is doing.
Then, make modifications and benchmark to understand what in the data pipeline is your bottleneck.
Do __NOT__ look at training speed when you benchmark a DataFlow, only use the output of `TestDataSpeed`.

A DataFlow could be blocked by CPU/disk/network/IPC bandwidth.
Do __NOT__ optimize the DataFlow before knowing what it is blocked on.
By benchmarking with modifications to your dataflow, you can see which
components is the bottleneck of your dataflow. For example, with a simple
dataflow, you can usually do the following:

1. If your dataflow becomes fast enough after removing some pre-processing (e.g.
   augmentations), then the pre-processing is the bottleneck.
1. Without pre-processing, your dataflow is just reading + parallelism, which
   includes both reading cost and the multiprocess communication cost.
   You can now let your reader produce only a single float after reading a large
   amount of data, so that the pipeline contains only parallel reading, but negligible
   communication cost any more. 
   
   If this becomes fast enough, it means that communication is the bottleneck.
   If pure parallel reading is still not fast enough, it means your raw reader is the bottleneck.
1. In practice the dataflow can be more complicated and you'll need to design
   your own strategies to understand its performance.
   
Once you've understand what is the bottleneck, you can try some improvements such as:

1. Use single-file database to avoid random read on hard disk.
2. Use fewer pre-processings or write faster ones with whatever tools you have.
3. Move certain pre-processing (e.g. mean/std normalization) to the graph, if TF has fast implementation of it.
4. Compress your data (e.g. use uint8 images, or JPEG-compressed images) before sending them through anything (network, ZMQ pipe, Python-TF copy etc.)
5. Use distributed data preprocessing, with `send_dataflow_zmq` and `RemoteDataZMQ`.

## Investigate TensorFlow

When you're sure that data is not a bottleneck (e.g. when the logs show that queue is almost full), you can start to
worry about the model.

A naive but effective way is to remove ops from your model to understand how much time they cost.
Or you can use `GraphProfiler` callback to benchmark the graph. It will
dump runtime tracing information (to either TensorBoard or chrome) to help diagnose the issue.
Remember not to use the first several iterations.

### Slow on single-GPU
This is literally saying TF ops are slow. Usually there isn't much you can do, except to optimize the kernels.
But there may be something cheap you can try:

1. Visualize copies across devices in chrome.
	 It may help to change device placement to avoid some CPU-GPU copies.
	 It may help to replace some CPU-only ops with equivalent GPU ops to avoid copies.

2. Sometimes there are several mathematically equivalent ways of writing the same model
	 with different ops and therefore different speed.

### Cannot scale to multi-GPU
If you're unable to scale to multiple GPUs almost linearly:
1. First make sure that the ImageNet-ResNet example can scale. Run it with `--fake` to use fake data.
	If not, it's a bug or an environment setup problem.
2. Then note that your model may have a different communication-computation pattern that affects efficiency.
	 There isn't a simple answer to this.
	 You may try a different multi-GPU trainer; the speed can vary a lot between
	 trainers in rare cases.

Note that scalibility is always measured by keeping "batch size per GPU" constant.
