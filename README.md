# tensorpack
Neural Network Toolbox on TensorFlow

In development but usable. API might change a bit.

See some interesting [examples](https://github.com/ppwwyyxx/tensorpack/tree/master/examples) to learn about the framework:

+ [DoReFa-Net: low bitwidth CNN](https://github.com/ppwwyyxx/tensorpack/tree/master/examples/DoReFa-Net)
+ [Double-DQN for playing Atari games](https://github.com/ppwwyyxx/tensorpack/tree/master/examples/Atari2600)
+ [ResNet for Cifar10 classification](https://github.com/ppwwyyxx/tensorpack/tree/master/examples/ResNet)
+ [char-rnn language model](https://github.com/ppwwyyxx/tensorpack/tree/master/examples/char-rnn)

## Features:

Focused on modularity. Just have to define the three components in training:

1. The model, or the graph. Define its input and output. `models/` has some scoped abstraction of common models.

2. The data. All data producer has a unified `DataFlow` interface, and this interface can be chained
	 to perform complex preprocessing. It uses multiprocess to avoid performance bottleneck.

3. The callbacks. They include everything you want to do besides the training iterations:
	change hyperparameters, save model, print logs, run validation, and more.
