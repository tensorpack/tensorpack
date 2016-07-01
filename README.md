# tensorpack
Neural Network Toolbox on TensorFlow

Still in development, but usable.

See some interesting [examples](https://github.com/ppwwyyxx/tensorpack/tree/master/examples) to learn about the framework:

+ [DoReFa-Net: low bitwidth CNN](https://github.com/ppwwyyxx/tensorpack/tree/master/examples/DoReFa-Net)
+ [Double-DQN for playing Atari games](https://github.com/ppwwyyxx/tensorpack/tree/master/examples/Atari2600)
+ [ResNet for Cifar10 classification](https://github.com/ppwwyyxx/tensorpack/tree/master/examples/ResNet)
+ [char-rnn language model](https://github.com/ppwwyyxx/tensorpack/tree/master/examples/char-rnn)

## Features:

Focused on modularity. Just have to define the three components in training:

1. The model, or the graph. Define the graph as well as its inputs and outputs. `models/` has some scoped abstraction of common models.

2. The data. All data producer has an unified `DataFlow` interface, and this interface can be chained
	 to perform complex preprocessing. It uses multiprocess to avoid performance bottleneck on data
	 loading.

3. The callbacks. They include everything you want to do apart from the training iterations:
	change hyperparameters, save models, print logs, run validation, and more.

With the above components defined, tensorpack trainer will run the training iterations for you.
Multi-GPU training is ready to use by simply changing the trainer.

## Dependencies:

+ Python 2 or 3
+ TensorFlow >= 0.8
+ Python bindings for OpenCV
+ other requirements:
```
pip install --user -r requirements.txt
pip install --user -r opt-requirements.txt (some optional dependencies)
```
+ allow `import tensorpack` everywhere:
```
export PYTHONPATH=$PYTHONPATH:`readlink -f path/to/tensorpack`
```


