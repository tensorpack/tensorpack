
# Trainer

Tensorpack follows the "define-and-run" paradigm. A training has two steps:

1. __Define__: Build graph for the model.
	Users can call whatever tensorflow functions to setup the graph.
	Users may or may not use tensorpack `InputSource`, `ModelDesc` or other utilities to build the graph.
	This goal of this step is to define "what to run" in later training steps,
	and it can happen either inside or outside tensorpack trainer.

2. __Run__: Train the model (the [Trainer.train() method](../modules/train.html#tensorpack.train.Trainer.train)):

	1. Setup callbacks/monitors.
	2. Finalize graph, initialize session.
	3. Run the training loop.


## Assumptions of Base Trainer

In research we do training of various kind.
Tensorpack trainers try to avoid making assumptions on what type of training
you want to do (e.g., it doesn't have to be batched, SGD-like, or have `X`(inputs) and `y`(outputs)).
The only assumption tensorpack `Trainer` class makes about your training, is that your training
follows this pattern:
```python
for epoch_num in range(starting_epoch, max_epoch):
	for local_step in range(steps_per_epoch):
		run_step()
```

1. Training is **running some iterations**.
Tensorpack base trainer implements the logic of __running the iteration__.
Users or derived trainers should implement __what the iteration is__.

2. Trainer assumes the existence of __"epoch"__, i.e. that the iterations run in double for-loops.
But the epoch size can actually be any number you set
and it only affects the [schedule of callbacks](extend/callback.html).
In other words, an "epoch" in tensorpack is the __default period to run callbacks__ (validation, summary, checkpoint, etc.).


### Single-Cost Trainers

Most neural network training tasks are single-cost optimization.
Tensorpack provides some trainer implementations for such tasks.
These trainers will take care of step 1, by building the graph by itself, with the following arguments:

1. Some `InputDesc`, the metadata about the input.
2. An `InputSource`, where the input come from. See [Input Pipeline](input-source.html).
3. A function which takes input tensors and returns the cost.
4. A function which returns an optimizer.

These are documented better in [SingleCostTrainer.setup_graph](../modules/train.html#tensorpack.train.SingleCostTrainer.setup_graph).
Often you'll not use this method directly, but use [high-level interface](training-interface.html#with-modeldesc-and-trainconfig)
instead.

Existing multi-GPU trainers include the logic of single-cost data-parallel training.
You can enable them by just one line, and all the necessary logic to achieve the best performance was baked into the trainers already.
The trainers can reach the same performance as the [official tensorflow benchmark](https://www.tensorflow.org/performance/benchmarks).

Please note that in data-parallel training, in each iteration all towers (all replicates of the model) will take
tensors from the `InputSource` (instead of taking one for all and split). So the total batch size
would be ``(batch size of InputSource/DataFlow) * #GPU``.

### Custom Trainers

You can easily write a trainer for other types of training.
See [Write a Trainer](extend/trainer.html).

