
# Trainer

Training is **running something again and again**.
Tensorpack base trainer implements the logic of *running the iteration*,
and other trainers implement *what the iteration is*.

Most neural network training tasks are single-cost optimization.
Tensorpack provides some trainer implementations for such tasks.
These trainers will by default minimizes `ModelDesc.cost`.
Therefore, you can use these trainers as long as you set `self.cost` in `ModelDesc._build_graph()`,
as most examples did.

Existing trainers were implemented with certain prefetch mechanism,
which will run significantly faster than a naive `sess.run(..., feed_dict={...})`.

There are also Multi-GPU trainers which include the logic of data-parallel Multi-GPU training.
You can enable them by just changing one line, and all the necessary logic to achieve the best
performance was baked into the trainers already.
For example, SyncMultiGPUTrainer can train ResNet50 as fast as the [official benchmark](https://github.com/tensorflow/benchmarks).

To use trainers, pass a `TrainConfig` to configure them:

```python
config = TrainConfig(
           model=MyModel()
           dataflow=my_dataflow,
           callbacks=[...]
         )

# start training (with a slow trainer. See 'tutorials - Input Pipeline' for details):
# SimpleTrainer(config).train()

# start training with queue prefetch:
QueueInputTrainer(config).train()

# start multi-GPU training with a synchronous update:
# SyncMultiGPUTrainer(config).train()
```

Trainers just run some iterations, so there is no limit to where the data come from
or what to do in an iteration.
For example, [GAN trainer](../examples/GAN/GAN.py) minimizes
two cost functions alternatively.
