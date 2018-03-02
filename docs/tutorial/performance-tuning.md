
# Performance Tuning

__We do not know why your training is slow__ (and most of the times it's not a tensorpack problem).
Performance is different across machines and tasks,
so you need to figure out most parts by your own.
Here's a list of things you can do when your training is slow.

If you ask for help understanding and improving the speed, PLEASE do them and include your findings.

## Figure out the bottleneck

1. If you use feed-based input (unrecommended) and datapoints are large, data is likely to become the bottleneck.
2. If you use queue-based input + dataflow, you can look for the queue size statistics in
	 training log. Ideally the input queue should be near-full (default size is 50).
 	 If the size is near-zero, data is the bottleneck.
3. If GPU utilization is low, it may be because of slow data, or some ops are inefficient. Also make sure GPUs are not locked in P8 state.

## Benchmark the components
1. (usually not needed) Use `data=DummyConstantInput(shapes)` for training,
	so that the iterations only take data from a constant tensor.
	This will benchmark the graph without the overhead of data.
2. Use `dataflow=FakeData(shapes, random=False)` to replace your original DataFlow by a constant DataFlow.
  This is almost the same as (1).
3. If you're using a TF-based input pipeline you wrote, you can simply run it in a loop and test its speed.
4. Use `TestDataSpeed(mydf).start()` to benchmark your DataFlow.

A benchmark will give you more precise information about which part you should improve.
Note that you should only look at iteration speed after about 50 iterations, since everything is slow at the beginning.

## Investigate DataFlow

Understand the [Efficient DataFlow](efficient-dataflow.html) tutorial, so you know what your DataFlow is doing.

Benchmark your DataFlow with modifications to understand which part is the bottleneck. Some examples include:

1. Benchmark only raw reader (and perhaps add some parallelism).
2. Gradually add some pre-processing and see how the performance changes.
3. Change the number of parallel processes or threads.

A DataFlow could be blocked by CPU/disk/network/IPC bandwidth. Only by benchmarking will you
know the reason and improve it accordingly, e.g.:

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

### Slow with single-GPU
This is literally saying TF ops are slow. Usually there isn't much you can do, except to optimize the kernels.
But there may be something cheap you can try:

1. Visualize copies across devices in chrome.
	 It may help to change device placement to avoid some CPU-GPU copies.
	 It may help to replace some CPU-only ops with equivalent GPU ops to avoid copies.

2. Sometimes there are several mathematically equivalent ways of writing the same model
	 with different ops and therefore different speed.

### Cannot scale to multi-GPU
If you're unable to scale to multiple GPUs almost linearly:
1. First make sure that the ResNet example can scale. Run it with `--fake` to use fake data.
	If not, it's a bug or an environment setup problem.
2. Then note that your model may have a different communication-computation pattern that affects efficiency.
	 There isn't a simple answer to this.
	 You may try a different multi-GPU trainer; the speed can vary a lot sometimes.

Note that scalibility is always measured by keeping "batch size per GPU" constant.
