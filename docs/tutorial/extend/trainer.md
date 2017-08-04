
## Write a Trainer

The existing trainers should be enough for single-cost optimization tasks.
If you want to do something different during training, first consider writing it as a callback,
or write an issue to see if there is a better solution than creating new trainers.

For certain tasks, you do need a new trainer.

Trainers just run __some__ iterations, so there is no limit in where the data come from or what to do in an iteration.
The existing common trainers do two things:
1. Setup the graph and input pipeline, from `TrainConfig`.
2. Minimize `model.cost` in each iteration.

But you can customize it by using the base `Trainer` class.

* To customize the graph:

  Create the graph, add any tensors and ops either before creating the trainer or inside `Trainer.__init__`.

* Two ways to customize the iteration:

	1. Set `Trainer.train_op`. This op will be run by default.
	2. Subclass `Trainer` and override the `run_step()` method. This way you can run more ops in one iteration.

There are several different [GAN trainers](../../examples/GAN/GAN.py) for reference.
