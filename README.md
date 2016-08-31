# tensorpack
Neural Network Toolbox on TensorFlow

Still in development, but usable.

See some interesting [examples](examples) to learn about the framework:

+ [DoReFa-Net: training binary / low bitwidth CNN](examples/DoReFa-Net)
+ [Double-DQN for playing Atari games](examples/Atari2600)
+ [ResNet for Cifar10 classification](examples/ResNet)
+ [IncpetionV3 on ImageNet](examples/Inception/inceptionv3.py)
+ [char-rnn language model](examples/char-rnn)

## Features:

Abstract your training task into three components:

1. Model, or graph. `models/` has some scoped abstraction of common models. This part is roughly an
	 equivalent of slim/tflearn/tensorlayer.
	`LinearWrap` and `argscope` makes large models look simpler.

2. Data. tensorpack allows and encourages complex data processing.

	+ All data producer has an unified `DataFlow` interface, allowing them to be composed to perform complex preprocessing.
	+ Use Python to easily handle your own data format, yet still keep a good training speed thanks to multiprocess prefetch & TF Queue prefetch.
	For example, InceptionV3 can run in the same speed as the official code which reads data using TF operators.

3. Callbacks, including everything you want to do apart from the training iterations. Such as:
	+ Change hyperparameters during training
	+ Print some variables of interest
	+ Run inference on a test dataset
	+ Run some operations once a while
	+ Send the accuracy to your phone

With the above components defined, tensorpack trainer will run the training iterations for you.
Multi-GPU training is off-the-shelf by simply switching the trainer.

## Dependencies:

+ Python 2 or 3
+ TensorFlow >= 0.8
+ Python bindings for OpenCV
+ other requirements:
```
pip install --user -r requirements.txt
pip install --user -r opt-requirements.txt (some optional dependencies, you can install later if needed)
```
+ Use [tcmalloc](http://goog-perftools.sourceforge.net/doc/tcmalloc.html) whenever possible
+ Enable `import tensorpack`:
```
export PYTHONPATH=$PYTHONPATH:`readlink -f path/to/tensorpack`
```
