
# Trainer

In research we do training of various kind.
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
But an epoch doesn't need to be a full pass of your dataset, the size of an epoch can be any number you set
and it only affects the [schedule of callbacks](extend/callback.html).
In other words, an "epoch" in tensorpack is the __default period to run callbacks__ (validation, summary, checkpoint, etc.).


### Common Trainers

Most neural network training tasks are single-cost optimization.
Tensorpack provides some trainer implementations for such tasks.
These trainers will build the graph based on the given `ModelDesc`, and minimizes `ModelDesc.cost`.

To use trainers, pass a `TrainConfig` to configure them:

```python
config = TrainConfig(
           model=MyModel()
           dataflow=my_dataflow,
           # data=my_inputsource, # alternatively, use a customized InputSource
           callbacks=[...]
         )

# start training:
SomeTrainer(config, other_arguments).train()

# start multi-GPU training with synchronous update:
# SyncMultiGPUTrainerParameterServer(config).train()
```

When you set the DataFlow (rather than the InputSource) in the config,
tensorpack trainers automatically adopt certain prefetch mechanism, as mentioned
in the [Input Pipeline](input-source.html) tutorial.
You can set the InputSource instead, to customize this behavior.

Existing multi-GPU trainers include the logic of data-parallel training.
You can enable them by just one line, and all the necessary logic to achieve the best performance was baked into the trainers already.
The trainers can reach the same performance as the [official tensorflow benchmark](https://www.tensorflow.org/performance/benchmarks).

Please note that in data-parallel training, in each iteration all towers (all replicates of the model) will take
tensors from the InputSource (instead of taking one for all and split). So the total batch size
would be ``(batch size of InputSource/DataFlow) * #GPU``.

### Custom Trainers

You can easily write a trainer for other types of training.
See [Write a Trainer](extend/trainer.html).

