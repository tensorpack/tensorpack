
# Trainer

Training is **running something again and again**.
Tensorpack base trainer implements the logic of __running the iteration__.
Users or derived trainers should implement __what the iteration is__.


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

# start multi-GPU training with a synchronous update:
# SyncMultiGPUTrainerParameterServer(config).train()
```

When you set the DataFlow (rather than the InputSource) in the config,
tensorpack trainers automatically pick up certain prefetch mechanism,
which will run faster than a naive `sess.run(..., feed_dict={...})`.
You can set the InputSource instead, to customize this behavior.

Existing multi-GPU trainers include the logic of data-parallel training.
You can enable them by just one line, and all the necessary logic to achieve the best performance was baked into the trainers already.
The trainers can reach the same performance as the [official tensorflow benchmark](https://github.com/tensorflow/benchmarks).

Please note that, in data-parallel training, all towers (all replicates of the model) will take 
tensors from the InputSource (instead of taking one for all and split). So the total batch size
would be multiplied by the number of GPUs.

### Custom Trainers

Trainers just run __some__ iterations, so there is no limit in where the data come from or what to do in an iteration.
The existing trainers implement the default logic, but you can implement them yourself by using the base `Trainer` class. 

* Two ways to customize the graph:

  1. Create the graph, add any tensors and ops before creating the trainer. 
	2. Subclass `Trainer` and override the `_setup()` method which will be called in `Trainer.__init__`.

* Two ways to customize the iteration:

	1. Set `Trainer.train_op`. This op will be run by default.
	2. Subclass `Trainer` and override the `run_step()` method.

There are several different [GAN trainers](../examples/GAN/GAN.py) for reference.
