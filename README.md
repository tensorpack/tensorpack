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
+ [Generative Adversarial Network(GAN) variants](examples/GAN), including DCGAN, InfoGAN, Conditional GAN, WGAN, Image to Image.


### Speech / NLP:
+ [LSTM-CTC for speech recognition](examples/CTC-TIMIT)
+ [char-rnn for fun](examples/Char-RNN)
+ [LSTM language model on PennTreebank](examples/PennTreebank)

The examples are not only for demonstration of the framework -- you can train them and reproduce the results in papers.

## Features:

Describe your training task with three components:

1. __DataFlow__. process data in Python, with ease and speed.

	+ Allows you to process data in Python without blocking the training, by multiprocess prefetch & TF Queue prefetch.
	+ All data producer has a unified interface, you can compose and reuse them to perform complex preprocessing.

2. __Callbacks__, customizable, like `tf.train.SessionRunHook` but more than that. Includes everything you want to do apart from the training iterations, such as:
	+ Change hyperparameters during training
	+ Print some tensors of interest
	+ Run inference on a test dataset
	+ Run some operations once a while
	+ Send loss to your phone

3. __Model__, or graph. `models/` has some scoped abstraction of common models, but you can just use
	 symbolic functions in tensorflow or slim/tflearn/tensorlayer/etc.
	`LinearWrap` and `argscope` simplify large models (e.g. [vgg example](https://github.com/ppwwyyxx/tensorpack/blob/master/examples/load-vgg16.py)).

With the above components defined, tensorpack trainer runs the training iterations for you.
Trainer was written with performance in mind:
Even on a small CNN example, the training runs [2x faster](https://gist.github.com/ppwwyyxx/8d95da79f8d97036a7d67c2416c851b6) than the equivalent Keras code.

Multi-GPU training is off-the-shelf by simply switching the trainer.
You can also define your own trainer for non-standard training (e.g. GAN).

## Install:

Dependencies:

+ Python 2 or 3
+ TensorFlow >= 1.0.0
+ Python bindings for OpenCV
```
pip install --user -U git+https://github.com/ppwwyyxx/tensorpack.git
```
