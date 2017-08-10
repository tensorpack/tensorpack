
## Write a Trainer

The existing trainers should be enough for single-cost optimization tasks.
If you want to do something different during training, first consider writing it as a callback,
or write an issue to see if there is a better solution than creating new trainers.
If your task is fundamentally different from single-cost optimization, you may need to write a trainer.

Trainers just run __some__ iterations, so there is no limit in where the data come from or what to do in an iteration.
The existing common trainers all implement two things:
1. Setup the graph and input pipeline, using the given `TrainConfig`.
2. Minimize `model.cost` in each iteration.

But you can customize it by using the base `Trainer` class.

* To customize the graph:

  Add any tensors and ops you like, either before creating the trainer or inside `Trainer.__init__`.
	In this case you don't need to set model/data in `TrainConfig` any more.

* Two ways to customize the iteration:

	1. Set `Trainer.train_op`. This op will be run by default.
	2. Subclass `Trainer` and override the `run_step()` method. This way you can do something more than running an op.

There are several different [GAN trainers](../../examples/GAN/GAN.py) for reference.
The implementation of [SimpleTrainer](https://github.com/ppwwyyxx/tensorpack/blob/master/tensorpack/train/simple.py) may also be helpful.
