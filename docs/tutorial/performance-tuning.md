
# Performance Tuning

__We do not know why your training is slow__.
Performance is different on every machine. So you need to figure out most parts by your own.
Here's a list of things you can do when your training is slow.

If you're going to open an issue about slow training, PLEASE do them and include your findings.

## Figure out the bottleneck

1. If you use feed-based input (unrecommended) and datapoints are large, data is likely to become the
	 bottleneck.
2. If you use queue-based input + dataflow, you can look for the queue size statistics in
	 training log. Ideally the queue should be near-full (default size is 50).
 	 If the size is near-zero, data is the bottleneck.
3. If the GPU utilization is low, it may be because of slow data, or some ops are inefficient. Also make sure GPUs are not locked in P8 state.

## Benchmark the components
1. Use `DummyConstantInput(shapes)` as the `InputSource`.
	so that the iterations doesn't take any data from Python side but train on a constant tensor.
	This will help find out the slow operations you're using in the graph.
2. Use `dataflow=FakeData(shapes, random=False)` to replace your original DataFlow by a constant DataFlow.
  This has similar effect to (1), i.e., it eliminates the overhead of data.
3. If you're using a TF-based input pipeline you wrote, you can simply run it in a loop and test its speed.
4. Use `TestDataSpeed(mydf).start()` to benchmark your DataFlow.

A benchmark will give you more precise information about which part you should improve.

## Investigate DataFlow

Understand the [Efficient DataFlow](efficient-dataflow.html) tutorial, so you know what your DataFlow is doing.

Benchmark your DataFlow with modifications and you'll understand why it runs slow. Some examples
include:

1. Remove everything except for the raw reader (and perhaps add some prefetching).
2. Remove some suspicious pre-processing.
3. Change the number of parallel processes or threads.

A DataFlow could be blocked by CPU/hard disk/network/IPC bandwidth. Only by benchmarking will you
know the reason and improve it accordingly, e.g.:

1. Use single-file database to avoid random read on hard disk.
2. Write faster pre-processing with whatever tools you have.
3. Move certain pre-processing (e.g. mean/std normalization) to the graph, if TF has fast implementation of it.
4. Compress your data (e.g. use uint8 images, or JPEG-compressed images) before sending them through
	 anything (network, ZMQ pipe, Python-TF copy etc.)
5. Use distributed data preprocessing, with `send_dataflow_zmq` and `RemoteDataZMQ`.

## Investigate TensorFlow

When you're sure that data is not a bottleneck (e.g. when queue is always full), you can start to
worry about the model.

You can add a `GraphProfiler` callback when benchmarking the graph. It will
dump runtime tracing information (to either TensorBoard or chrome) to help diagnose the issue.

### Slow with single-GPU
This is literally saying TF ops are slow. Usually there isn't much you can do, except to optimize the kernels.
But there may be something cheap you can try:

1. You can visualize copies across devices in chrome.
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

Note that scalibility is always measured with the same "batch size per GPU", not the same total equivalent batch size.
