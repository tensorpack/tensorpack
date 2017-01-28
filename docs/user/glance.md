
## A High-Level Glance

The following guide introduces some core concepts of TensorPack. In contrast to several other libraries TensorPack contains of several modules to build complex deep learning algorithms and train models with high accuracy and high speed.

### Training
Given TensorFlow's optimizers this library provides several training protocols even for efficient multi-GPU environments. There is support for single GPU, training on one machine with multiple GPUs (synchron or asyncron), training of Generative Adversarial networks and reinforcement learning.

You only need to configure your training protocol like

````python
config =  TrainConfig(
            dataflow=my_dataflow,
            optimizer=tf.train.AdamOptimizer(lr),
            callbacks=Callbacks([ModelSaver(), ...]),
            model=Model())

# start training
SimpleTrainer(config).train()
````

Switching between single-GPU and multi-GPU is as easy as replace the last line with

````python
# start multi-GPUtraining
SyncMultiGPUTrainer(config).train()
````

### Callbacks

The use of callbacks add the flexibility to execute code during training. These callbacks are triggered on several events such as after each step or at the end of one training epoch.
