
# Trainer

Training is **running something again and again**.
Tensorpack base trainer implements the logic of *running the iteration*,
and other trainers implement *what the iteration is*.

Most neural network training tasks are single-cost optimization.
Tensorpack provides some trainer implementations for such tasks.
These trainers will by default minimizes `ModelDesc.cost`.
Therefore, you can use these trainers as long as you set `self.cost` in `ModelDesc._build_graph()`,
as most examples did.

Most existing trainers were implemented with a TensorFlow queue to prefetch and buffer
training data, which is faster than a naive `sess.run(..., feed_dict={...})`.
There are also multi-GPU trainers which include the logic of data-parallel multi-GPU training,
with either synchronous update or asynchronous update. You can enable multi-GPU training
by just changing one line.

To use trainers, pass a `TrainConfig` to configure them:

```python
config = TrainConfig(
           model=MyModel()
           dataflow=my_dataflow,
           callbacks=[...]
         )

# start training:
# SimpleTrainer(config).train()

# start training with queue prefetch:
# QueueInputTrainer(config).train()

# start multi-GPU training with a synchronous update:
SyncMultiGPUTrainer(config).train()
```

Trainers just run some iterations, so there is no limit to where the data come from
or what to do in an iteration.
For example, [GAN trainer](../examples/GAN/GAN.py) minimizes
two cost functions alternatively.
Some trainer takes data from a TensorFlow reading pipeline instead of a Dataflow
([PTB example](../examples/PennTreebank)).
