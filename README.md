# tensorpack
Neural Network Toolbox on TensorFlow

Still in development. Underlying design may change.

See some [examples](examples) to learn about the framework.
You can actually train them and reproduce the performance... not just to see how to write code.

+ [DoReFa-Net: training binary / low bitwidth CNN](examples/DoReFa-Net)
+ [ResNet for ImageNet/Cifar10/SVHN classification](examples/ResNet)
+ [InceptionV3 on ImageNet](examples/Inception/inceptionv3.py)
+ [Fully-convolutional Network for Holistically-Nested Edge Detection](examples/HED)
+ [Spatial Transformer Networks on MNIST addition](examples/SpatialTransformer)
+ [Deep Convolutional Generative Adversarial Networks](examples/GAN)
+ [DQN variants on Atari games](examples/Atari2600)
+ [Asynchronous Advantage Actor-Critic(A3C) with demos on OpenAI Gym](examples/OpenAIGym)
+ [char-rnn language model](examples/char-rnn)

## Features:

Describe your training task with three components:

1. Model, or graph. `models/` has some scoped abstraction of common models, but you can simply use
	 anything available in tensorflow. This part is roughly an equivalent of slim/tflearn/tensorlayer.
	`LinearWrap` and `argscope` makes large models look simpler.

2. Data. tensorpack allows and encourages complex data processing.

	+ All data producer has an unified `generator` interface, allowing them to be composed to perform complex preprocessing.
	+ Use Python to easily handle any of your own data format, yet still keep a good training speed thanks to multiprocess prefetch & TF Queue prefetch.
	For example, InceptionV3 can run in the same speed as the official code which reads data using TF operators.

3. Callbacks, including everything you want to do apart from the training iterations, such as:
	+ Change hyperparameters during training
	+ Print some variables of interest
	+ Run inference on a test dataset
	+ Run some operations once a while
	+ Send the accuracy to your phone

With the above components defined, tensorpack trainer will run the training iterations for you.
Multi-GPU training is off-the-shelf by simply switching the trainer.

## Dependencies:

+ Python 2 or 3
+ TensorFlow >= 0.10
+ Python bindings for OpenCV
+ other requirements:
```
pip install --user -r requirements.txt
pip install --user -r opt-requirements.txt (some optional dependencies, you can install later if needed)
```
+ [tcmalloc](http://goog-perftools.sourceforge.net/doc/tcmalloc.html) usually helps.
+ Enable `import tensorpack`:
```
export PYTHONPATH=$PYTHONPATH:`readlink -f path/to/tensorpack`
```
