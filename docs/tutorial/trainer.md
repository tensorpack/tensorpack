# Trainers

TensorFlow & Tensorpack follow the "define-and-run" paradigm.
Therefore a training contains two steps:

1. __Define__: Build graph for the model.
	Users can call whatever tensorflow functions to setup the graph.
	Users may or may not use tensorpack `InputSource`, `ModelDesc` or other utilities to build the graph.
	The goal of this step is to define "what to run" in later training steps.

2. __Run__: Train the model (the [Trainer.train() method](/modules/train.html#tensorpack.train.Trainer.train)):

	1. Setup callbacks/monitors.
	2. Finalize graph, initialize session.
	3. Run the training loop.

Tensorpack `Trainers` aims to simplify the above two steps
by exploiting some universal patterns.

### Assumptions of Base Trainer

* Q: What types of training can you do with tensorpack?
* A: Anything that runs in a loop.

In research we do training of various kind.
Tensorpack trainers avoid making assumptions on what type of training
you want to do. For example, unlike Keras, tensorpack does not wrongly assume that:
1. Your training data is batched
2. Your training is gradient-based optimization
3. Your data has `X`(inputs) and `y`(outputs)
4. You want to evaluate on zero or one validation dataset
5. ... and more

The only assumption is that your training follows this pattern:
```python
for epoch_num in range(starting_epoch, max_epoch):
	for local_step in range(steps_per_epoch):
		run_step()  # do something
```

In other words, the assumptions are:
1. Training is **running some iterations**.
Tensorpack base trainer implements the logic of __running the iterations__.
Users or derived trainers should implement __what the iterations are__.

2. The concept of __"epoch"__, i.e. we assume that the iterations run in nested for-loops.
In fact, the steps per epoch can be any number
and it only affects the [schedule of callbacks](./callback.md).
In other words, an "epoch" in tensorpack is the __default period to run
callbacks__ (validation, summary, checkpoint, etc.).
So this assumption effectively puts no extra constraints.


### Built-in Trainers

Tensorpack implements a few builtin trainers for __single-cost gradient-based optimization__,
as this is the most common type of task.
If your training follows this pattern, you only need to __select a trainer__,
and use it with its [training interface](./training-interface.md).

The simplest example of such a trainer is
[SimpleTrainer](../modules/train.html#tensorpack.train.SimpleTrainer).
All it does is building your model (which you have to provide) once
(or twice if inference is needed by callbacks) and minimizing its cost.

### Multi-GPU Trainers

For data-parallel multi-GPU training, different [multi-GPU trainers](../modules/train)
implement different distribution strategies.
They take care of device placement, gradient averaging and synchronization
in the efficient way, which is why multi-GPU training in tensorpack
is up to
[5x faster than Keras](https://github.com/tensorpack/benchmarks/tree/master/other-wrappers).
It takes only one line of code change to use them, e.g. `trainer=SyncMultiGPUTrainerReplicated(...)`.

Note some __common confusions__ when using these trainers:

1. In each iteration, instead of taking one input tensor for all GPUs and split,
    tensorpack trainers let all GPUs take tensors from the input.
	Therefore, the total batch size across all GPUs is ``(batch size of input source) * #GPU``.
    You may want to change `steps_per_epoch` or learing rate appropriately according
    to the total batch size.

    ```eval_rst
    .. note::

        Splitting a tensor for data-parallel training (as done by frameworks like Keras)
        makes no sense at all.
        First, it wastes time doing the split because typically data is first concatenated by the user.
        Second, this puts unnecessary shape constraints on the data, that the
        inputs on each GPU needs to have compatible shapes.
    ```

2. The tower function (your model code) will get called once on each GPU.
   So you must follow some [rules of tower function](extend/trainer.html#rules-of-tower-function).

### Distributed Trainers

Distributed training needs the [horovod](https://github.com/horovod/horovod) library which offers high-performance allreduce implementation.
To run distributed training, first install horovod properly, then refer to the
documentation of [HorovodTrainer](../modules/train.html#tensorpack.train.HorovodTrainer).

Tensorpack has implemented some other distributed trainers using TF's native API,
but TensorFlow is not actively supporting its distributed training features, and
its native distributed performance isn't very good even today.
Therefore those trainers are not maintained and are __not recommended for use__.
