# tensorpack
Neural Network Toolbox on TensorFlow

[![Build Status](https://travis-ci.org/ppwwyyxx/tensorpack.svg?branch=master)](https://travis-ci.org/ppwwyyxx/tensorpack)
[![badge](https://readthedocs.org/projects/pip/badge/?version=latest)](http://tensorpack.readthedocs.io/en/latest/index.html)

Tutorials are not fully finished. See some [examples](examples) to learn about the framework:

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
+ [Generative Adversarial Network(GAN) variants](examples/GAN), including DCGAN, InfoGAN, Conditional GAN, Image to Image.


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

	+ All data producer has an unified interface, allowing them to be composed to perform complex preprocessing.
	+ Use Python to easily handle any data format, yet still keep good performance thanks to multiprocess prefetch & TF Queue prefetch.
	For example, InceptionV3 can run in the same speed as the official code which reads data by TF operators.

3. __Callbacks__, including everything you want to do apart from the training iterations, such as:
	+ Change hyperparameters during training
	+ Print some tensors of interest
	+ Run inference on a test dataset
	+ Run some operations once a while
	+ Send loss to your phone

With the above components defined, tensorpack trainer will run the training iterations for you.
Multi-GPU training is off-the-shelf by simply switching the trainer.
You can also define your own trainer for non-standard training (e.g. GAN).

## Install:

Dependencies:

+ Python 2 or 3
+ TensorFlow >= 1.0.0rc0
+ Python bindings for OpenCV
+ (optional) use tcmalloc if running with large data

```
pip install --user -U git+https://github.com/ppwwyyxx/tensorpack.git
pip install --user -r opt-requirements.txt # (some optional dependencies required by certain submodules, you can install later if prompted)
```
