
# Trainers

## Trainer

Training is basically **running something again and again**.
Tensorpack base trainer implements the logic of *running the iteration*,
and other trainers implement *what the iteration is*.

Most neural network training tasks are single-cost optimization.
Tensorpack provides some trainer implementations for such tasks.
These trainers will by default minimizes `ModelDesc.cost`,
therefore you can use these trainers as long as you set `self.cost` in `ModelDesc._build_graph()`,
as did in most examples.

Most existing trainers were implemented with a TensorFlow queue to prefetch and buffer
training data, which is significantly faster than
a naive `sess.run(..., feed_dict={...})`.
There are also multi-GPU trainers which includes the logic of data-parallel multi-GPU training,
with either synchronous update or asynchronous update. You can enable multi-GPU training
by just changing one line.

To use trainers, pass a `TrainConfig` to configure them:

````python
config = TrainConfig(
           dataflow=my_dataflow,
           optimizer=tf.train.AdamOptimizer(0.01),
           callbacks=[...]
           model=MyModel()
         )

# start training:
# SimpleTrainer(config).train()

# start training with queue prefetch:
# QueueInputTrainer(config).train()

# start multi-GPU training with synchronous update:
SyncMultiGPUTrainer(config).train()
````

Trainers just run some iterations, so there is no limit in where the data come from
or what to do in an iteration.
For example, [GAN trainer](../examples/GAN/GAN.py) minimizes
two cost functions alternatively.
Some trainer takes data from a TensorFlow reading pipeline instead of a Dataflow
([PTB example](../examples/PennTreebank)).


## Write a trainer

The existing trainers should be enough for single-cost optimization tasks. If you
want to do something inside the trainer, consider writing it as a callback, or
write an issue to see if there is a better solution than creating new trainers.

For other tasks, you might need a new trainer.
The [GAN trainer](../examples/GAN/GAN.py) is one example of how to implement
new trainers.

More details to come.
