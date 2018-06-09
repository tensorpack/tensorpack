## Understand Trainer

### Role of Trainer

Tensorpack follows the "define-and-run" paradigm. Therefore a training script has two steps:

1. __Define__: Build graph for the model.
	Users can call whatever tensorflow functions to setup the graph.
	Users may or may not use tensorpack `InputSource`, `ModelDesc` or other utilities to build the graph.
	The goal of this step is to define "what to run" in later training steps,
	and it can happen __either inside or outside__ tensorpack trainer.

2. __Run__: Train the model (the [Trainer.train() method](../modules/train.html#tensorpack.train.Trainer.train)):

	1. Setup callbacks/monitors.
	2. Finalize graph, initialize session.
	3. Run the training loop.


### Assumptions of Base Trainer

* Q: What types of training can you do with tensorpack?
* A: Anything that runs in a loop.

In research we do training of various kind.
Tensorpack trainers avoid making assumptions on what type of training
you want to do (e.g., it doesn't have to be batched, SGD-like, or have `X`(inputs) and `y`(outputs)).
The only assumption is that your training follows this pattern:
```python
for epoch_num in range(starting_epoch, max_epoch):
	for local_step in range(steps_per_epoch):
		run_step()
```

1. Training is **running some iterations**.
Tensorpack base trainer implements the logic of __running the iteration__.
Users or derived trainers should implement __what the iteration is__.

2. Trainer assumes the existence of __"epoch"__, i.e. that the iterations run in double for-loops.
But `steps_per_epoch` can be any number you set
and it only affects the [schedule of callbacks](extend/callback.html).
In other words, an "epoch" in tensorpack is the __default period to run callbacks__ (validation, summary, checkpoint, etc.).


### How Existing (Single-Cost) Trainers Work

Most neural network training tasks are single-cost optimization.
Tensorpack provides some trainer implementations for such tasks.
These trainers will take care of step 1 (define the graph), with the following arguments:

1. Some `InputDesc`, the metadata about the input.
2. An `InputSource`, where the input come from. See [Input Pipeline](input-source.html).
3. A function which takes input tensors and returns the cost.
4. A function which returns an optimizer.

These are documented in [SingleCostTrainer.setup_graph](../modules/train.html#tensorpack.train.SingleCostTrainer.setup_graph).
In practice you'll not use this method directly, but use [high-level interface](training-interface.html#with-modeldesc-and-trainconfig) instead.


### Write a Trainer

The existing trainers should be enough for data-parallel single-cost optimization tasks.
If you just want to do some extra work during training, first consider writing it as a callback,
or write an issue to see if there is a better solution than creating new trainers.
If your task is fundamentally different from single-cost optimization, you will need to write a trainer.

You can customize the trainer by either using or inheriting the base `Trainer` class.
You will need to do two things for a new Trainer:

1. Define the graph. There are 2 ways you can do this:
    1. Create any tensors and ops you need, before creating the trainer.
    2. Create them inside `Trainer.__init__`.

2. Define what is the iteration. There are 2 ways to define the iteration:
	1. Set `Trainer.train_op` to a TensorFlow operation. This op will be run by default.
	2. Subclass `Trainer` and override the `run_step()` method. This way you can
       do something more than running an op. 

       Note that trainer has `self.sess` and `self.hooked_sess`: only the hooked
       session will trigger the `before_run`/`after_run` callbacks.
       If you need more than one `Session.run` in one steps, special care needs
       to be taken to choose which session to use, because many states 
       (global steps, StagingArea, summaries) are maintained through `before_run`/`after_run`.
       

There are several different [GAN trainers](../../examples/GAN/GAN.py) for reference.
