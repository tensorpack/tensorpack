# Efficient DataFlow

This tutorial gives an overview of how to build an efficient DataFlow, using ImageNet
dataset as an example.
Our goal in the end is to have
a __generator__ which yields preprocessed ImageNet images and labels as fast as possible.
Since it is simply a generator interface, you can use the DataFlow in other frameworks (e.g. Keras)
or your own code as well.

We use ILSVRC12 training set, which contains 1.28 million images.
The original images (JPEG compressed) are 140G in total.
The average resolution is about 400x350 <sup>[[1]]</sup>.
Following the [ResNet example](../examples/ResNet), we need images in their original resolution,
so we will read the original dataset instead of a down-sampled version, and
apply complicated preprocessing to it.
We will need to reach a speed of, roughly 1000 images per second, to keep GPUs busy.

Note that the actual performance would depend on not only the disk, but also
memory (for caching) and CPU (for data processing).
You will need to tune the parameters (#processes, #threads, size of buffer, etc.)
or change the pipeline for new tasks and new machines to achieve the best performance.

This tutorial is quite complicated because you do need this knowledge of hardware & system to run fast on ImageNet-sized dataset.
However, for __small datasets__ (e.g., several GBs), a proper prefetch should work well enough.

## Random Read

We start from a simple DataFlow:
```python
from tensorpack import *
ds0 = dataset.ILSVRC12('/path/to/ILSVRC12', 'train', shuffle=True)
ds1 = BatchData(ds0, 256, use_list=True)
TestDataSpeed(ds1).start_test()
```

Here `ds0` simply reads original images from the filesystem. It is implemented simply by:
```python
for filename, label in filelist:
  yield [cv2.imread(filename), label]
```

And `ds1` batch the datapoints from `ds0`, so that we can measure the speed of this DataFlow in terms of "batch per second".
By default, `BatchData`
will stack the datapoints into an `numpy.ndarray`, but since images are original of different shapes, we use
`use_list=True` so that it just produces lists.

On an SSD you probably can already observe good speed here (e.g. 5 it/s, that is 1280 samples/s), but on HDD the speed may be just 1 it/s,
because we are doing heavy random read on the filesystem (regardless of whether `shuffle` is True).

We will now add the cheapest pre-processing now to get an ndarray in the end instead of a list
(because TensorFlow will need ndarray eventually):
```eval_rst
.. code-block:: python
	  :emphasize-lines: 2,3

		ds = dataset.ILSVRC12('/path/to/ILSVRC12', 'train', shuffle=True)
		ds = AugmentImageComponent(ds, [imgaug.Resize(224)])
		ds = BatchData(ds, 256)
```
You'll start to observe slow down after adding more pre-processing (such as those in the [ResNet example](../examples/ResNet/imagenet-resnet.py)).
Now it's time to add threads or processes:
```eval_rst
.. code-block:: python
	  :emphasize-lines: 3

		ds0 = dataset.ILSVRC12('/path/to/ILSVRC12', 'train', shuffle=True)
		ds1 = AugmentImageComponent(ds0, lots_of_augmentors)
		ds = PrefetchDataZMQ(ds1, nr_proc=25)
		ds = BatchData(ds, 256)
```
Here we started 25 processes to run `ds1`, and collect their output through ZMQ IPC protocol.
Using ZMQ to transfer data is faster than `multiprocessing.Queue`, but data copy (even
within one process) can still be quite expensive when you're dealing with large data.
For example, to reduce copy overhead, the ResNet example deliberately moves certain pre-processing (the mean/std normalization) from DataFlow to the graph.
This way the DataFlow only transfers uint8 images as opposed float32 which takes 4x more memory.

Alternatively, you can use multi-threading like this:
```python
ds0 = dataset.ILSVRC12('/path/to/ILSVRC12', 'train', shuffle=True)
augmentor = AugmentorList(lots_of_augmentors)
ds1 = ThreadedMapData(
    ds0, nr_thread=25,
    map_func=lambda x: augmentor.augment(x), buffer_size=1000)
# ds1 = PrefetchDataZMQ(ds1, nr_proc=1)
ds = BatchData(ds1, 256)
```
Since no `fork()` is happening here, there'll be only one instance of `ds0`.
25 threads will fetch data from `ds0`, run the augmentor function and
put results into a buffer of size 1000.
To reduce the effect of GIL, you can then uncomment the line so that everything above it (including all the
threads) happen in an independent process.

There is no answer whether it is faster to use threads or processes.
Processes avoid the cost of GIL but bring the cost of communication.
You can also try a combination of both (several processes each with several threads).


## Sequential Read

Random read is usually not a good idea, especially if the data is not on a SSD.
We can also dump the dataset into one single file and read it sequentially.

```python
from tensorpack import *
class RawILSVRC12(DataFlow):
    def __init__(self):
        meta = dataset.ILSVRCMeta()
        self.imglist = meta.get_image_list('train')
        # we apply a global shuffling here because later we'll only use local shuffling
        np.random.shuffle(self.imglist)
        self.dir = os.path.join('/path/to/ILSVRC', 'train')
    def get_data(self):
        for fname, label in self.imglist:
            fname = os.path.join(self.dir, fname)
            with open(fname, 'rb') as f:
                jpeg = f.read()
            jpeg = np.asarray(bytearray(jpeg), dtype='uint8')
            yield [jpeg, label]
    def size(self):
        return len(self.imglist)
ds0 = RawILSVRC12()
ds1 = PrefetchDataZMQ(ds0, nr_proc=1)
dftools.dump_dataflow_to_lmdb(ds1, '/path/to/ILSVRC-train.lmdb')
```
The above script builds a DataFlow which produces jpeg-encoded ImageNet data.
We store the jpeg string as a numpy array because the function `cv2.imdecode` expect it later.
We use 1 prefetch process to speed up. If `nr_proc>1`, `ds1` will take data
from several forks of `ds0` and will not be identical to `ds0` any more.

It will generate a database file of 140G. We build a DataFlow to read the LMDB file sequentially:
```
from tensorpack import *
ds = LMDBData('/path/to/ILSVRC-train.lmdb', shuffle=False)
ds = BatchData(ds, 256, use_list=True)
TestDataSpeed(ds).start_test()
```
Depending on whether the OS has cached the file for you (and how large the RAM is), the above script
can run at a speed of 10~130 it/s, roughly corresponding to 250MB~3.5GB/s bandwidth. You can test
your cached and uncached disk read bandwidth with `sudo hdparm -Tt /dev/sdX`.
As a reference, on Samsung SSD 850, the uncached speed is about 16it/s.

```eval_rst
.. code-block:: python
	  :emphasize-lines: 2

	  ds = LMDBData('/path/to/ILSVRC-train.lmdb', shuffle=False)
	  ds = LocallyShuffleData(ds, 50000)
	  ds = BatchData(ds, 256, use_list=True)
```
Instead of shuffling all the training data in every epoch (which would require random read),
the added line above maintains a buffer of datapoints and shuffle them once a while.
It will not affect the model as long as the buffer is large enough,
but it can also consume much memory if too large.

Then we add necessary transformations:
```eval_rst
.. code-block:: python
    :emphasize-lines: 3-5

    ds = LMDBData(db, shuffle=False)
    ds = LocallyShuffleData(ds, 50000)
    ds = LMDBDataPoint(ds)
    ds = MapDataComponent(ds, lambda x: cv2.imdecode(x, cv2.IMREAD_COLOR), 0)
    ds = AugmentImageComponent(ds, lots_of_augmentors)
    ds = BatchData(ds, 256)
```

1. `LMDBDataPoint` deserialize the datapoints (from string to [jpeg_string, label])
2. Use OpenCV to decode the first component into ndarray
3. Apply augmentations to the ndarray

Both imdecode and the augmentors can be quite slow. We can parallelize them like this:
```eval_rst
.. code-block:: python
    :emphasize-lines: 3,7

    ds = LMDBData(db, shuffle=False)
    ds = LocallyShuffleData(ds, 50000)
    ds = PrefetchData(ds, 5000, 1)
    ds = LMDBDataPoint(ds)
    ds = MapDataComponent(ds, lambda x: cv2.imdecode(x, cv2.IMREAD_COLOR), 0)
    ds = AugmentImageComponent(ds, lots_of_augmentors)
    ds = PrefetchDataZMQ(ds, 25)
    ds = BatchData(ds, 256)
```

Since we are reading the database sequentially, having multiple identical instances of the
underlying DataFlow will result in biased data distribution. Therefore we use `PrefetchData` to
launch the underlying DataFlow in one independent process, and only parallelize the transformations.
(`PrefetchDataZMQ` is faster but not fork-safe, so the first prefetch has to be `PrefetchData`. This is [issue#138](https://github.com/ppwwyyxx/tensorpack/issues/138))

Let me summarize what the above DataFlow does:

1. One process reads LMDB file, shuffle them in a buffer and put them into a `multiprocessing.Queue` (used by `PrefetchData`).
2. 25 processes take items from the queue, decode and process them into [image, label] pairs, and
	 send them through ZMQ IPC pipes.
3. The main process takes data from the pipe and feeds it into the graph, according to
	 how the `Trainer` is implemented.

The above DataFlow can run at a speed of 5~10 batches per second if you have good CPUs, RAM, disks and augmentors.
As a reference, tensorpack can train ResNet-18 (a shallow ResNet) at 4.5 batches (of 256 samples) per second on 4 old TitanX.
So DataFlow will not be a serious bottleneck if configured properly.

## More Efficient DataFlow

To work with larger datasets (or smaller networks, or more GPUs) you could be severely bounded by CPU or disk speed of a single machine.
One way is to optimize the preprocessing routine (e.g. write something in C++ or use TF reading operators).
Another way to scale is to run DataFlow in a distributed fashion and collect them on the
training machine. E.g.:
```python
# Data Machine #1, process 1-20:
df = MyLargeData()
send_dataflow_zmq(df, 'tcp://1.2.3.4:8877')
```
```python
# Data Machine #2, process 1-20:
df = MyLargeData()
send_dataflow_zmq(df, 'tcp://1.2.3.4:8877')
```
```python
# Training Machine, process 1-10:
df = MyLargeData()
send_dataflow_zmq(df, 'ipc:///tmp/ipc-socket')
```
```python
# Training Machine, training process
df = RemoteDataZMQ('ipc:///tmp/ipc-socket', 'tcp://0.0.0.0:8877')
TestDataSpeed(df).start_test()
```


[1]: #ref

<div id=ref> </div>

[[1]]. [ImageNet: A Large-Scale Hierarchical Image Database](http://www.image-net.org/papers/imagenet_cvpr09.pdf), CVPR09
