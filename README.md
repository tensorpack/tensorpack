# tensorpack
Neural Network Toolbox on TensorFlow

[![Build Status](https://travis-ci.org/ppwwyyxx/tensorpack.svg?branch=master)](https://travis-ci.org/ppwwyyxx/tensorpack)

Docs & tutorials should be ready within a month. See some [examples](examples) to learn about the framework:

### Vision:
+ [DoReFa-Net: train binary / low-bitwidth CNN on ImageNet](examples/DoReFa-Net)
+ [Train ResNet on ImageNet / Cifar10 / SVHN](examples/ResNet)
+ [InceptionV3 on ImageNet](examples/Inception/inceptionv3.py)
+ [Fully-convolutional Network for Holistically-Nested Edge Detection(HED)](examples/HED)
+ [Spatial Transformer Networks on MNIST addition](examples/SpatialTransformer)
+ [Visualize Saliency Maps by Guided ReLU](examples/Saliency)
+ [Similarity Learning on MNIST](examples/SimilarityLearning)

### Reinforcement Learning:
+ [Deep Q-Network(DQN) variants on Atari games](examples/DeepQNetwork)
+ [Asynchronous Advantage Actor-Critic(A3C) with demos on OpenAI Gym](examples/A3C-Gym)

### Unsupervised Learning:
+ [Generative Adversarial Network(GAN) variants, including DCGAN, Image2Image, InfoGAN](examples/GAN)

### Speech / NLP:
+ [LSTM-CTC for speech recognition](examples/CTC-TIMIT)
+ [char-rnn for fun](examples/Char-RNN)
+ [LSTM language model on PennTreebank](examples/PennTreebank)

The examples are not only for demonstration of the framework -- you can train them and reproduce the results in papers.

## Features:

Describe your training task with three components:

1. __Model__, or graph. `models/` has some scoped abstraction of common models, but you can simply use
	 any symbolic functions available in tensorflow, or most functions in slim/tflearn/tensorlayer.
	`LinearWrap` and `argscope` simplify large models ([vgg example](https://github.com/ppwwyyxx/tensorpack/blob/master/examples/load-vgg16.py)).

2. __DataFlow__. tensorpack allows and encourages complex data processing.

	+ All data producer has an unified `generator` interface, allowing them to be composed to perform complex preprocessing.
	+ Use Python to easily handle any data format, yet still keep good performance thanks to multiprocess prefetch & TF Queue prefetch.
	For example, InceptionV3 can run in the same speed as the official code which reads data by TF operators.

3. __Callbacks__, including everything you want to do apart from the training iterations, such as:
	+ Change hyperparameters during training
	+ Print some variables of interest
	+ Run inference on a test dataset
	+ Run some operations once a while
	+ Send loss to your phone

With the above components defined, tensorpack trainer will run the training iterations for you.
Multi-GPU training is off-the-shelf by simply switching the trainer.
You can also define your own trainer for non-standard training (e.g. GAN).

The components are designed to be independent. You can use Model or DataFlow in other projects as well.

## Dependencies:

+ Python 2 or 3
+ TensorFlow >= 1.0.0rc0
+ Python bindings for OpenCV
+ other requirements:
```
pip install --user -r requirements.txt
pip install --user -r opt-requirements.txt # (some optional dependencies required by certain submodule, you can install later if needed)
```
+ Enable `import tensorpack`:
```
export PYTHONPATH=$PYTHONPATH:`readlink -f path/to/tensorpack`
```
(or use `greadlink` from `coreutils` brew package if you're on OSX)

+ Use tcmalloc if running with large data
