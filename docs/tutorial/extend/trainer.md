
## Write a Trainer

The existing trainers should be enough for single-tower single-cost optimization tasks.
If you just want to do some extra work during training, first consider writing it as a callback,
or write an issue to see if there is a better solution than creating new trainers.
If your task is fundamentally different from single-cost optimization, you will need to write a trainer.


Trainers just run __some__ iterations, so there is no limit in where the data come from or what to do in an iteration.
The existing common trainers all implement two things:
1. Setup the graph and input pipeline, using the given `InputSource` and `get_cost_fn`.
2. Minimize `model.cost` in each iteration.

But you can customize it by using or inheriting the base `Trainer` class.
You will need to define two things for a new Trainer:

1. What is the graph.
	Add any tensors and ops you like, either before creating the trainer or inside `Trainer.__init__`.

* What is the iteration. There are 2 ways to define an iteration:
	1. Set `Trainer.train_op`. This op will be run by default.
	2. Subclass `Trainer` and override the `run_step()` method. This way you can do something more than running an op.

There are several different [GAN trainers](../../examples/GAN/GAN.py) for reference.
