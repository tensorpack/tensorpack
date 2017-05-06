# tensorpack
Neural Network Toolbox on TensorFlow.

[![Build Status](https://travis-ci.org/ppwwyyxx/tensorpack.svg?branch=master)](https://travis-ci.org/ppwwyyxx/tensorpack)
[![badge](https://readthedocs.org/projects/pip/badge/?version=latest)](http://tensorpack.readthedocs.io/en/latest/index.html)

See some [examples](examples) to learn about the framework:

### Vision:
+ [DoReFa-Net: train binary / low-bitwidth CNN on ImageNet](examples/DoReFa-Net)
+ [Train ResNet on ImageNet / Cifar10 / SVHN](examples/ResNet)
+ [Generative Adversarial Network(GAN) variants](examples/GAN), including DCGAN, InfoGAN, Conditional GAN, WGAN, BEGAN, DiscoGAN, Image to Image.
+ [Fully-convolutional Network for Holistically-Nested Edge Detection(HED)](examples/HED)
+ [Spatial Transformer Networks on MNIST addition](examples/SpatialTransformer)
+ [Visualize Saliency Maps by Guided ReLU](examples/Saliency)
+ [Similarity Learning on MNIST](examples/SimilarityLearning)

### Reinforcement Learning:
+ [Deep Q-Network(DQN) variants on Atari games](examples/DeepQNetwork), including DQN, DoubleDQN, DuelingDQN.
+ [Asynchronous Advantage Actor-Critic(A3C) with demos on OpenAI Gym](examples/A3C-Gym)

### Speech / NLP:
+ [LSTM-CTC for speech recognition](examples/CTC-TIMIT)
+ [char-rnn for fun](examples/Char-RNN)
+ [LSTM language model on PennTreebank](examples/PennTreebank)

The examples are not only for demonstration of the framework -- you can train them and reproduce the results in papers.

## Features:

It's Yet Another TF wrapper, but different in:
1. Not focus on models.
	+ There are already too many symbolic function wrappers.
		Tensorpack includes only a few common models, and helpful tools such as `LinearWrap` to simplify large models.
	  But you can use any other wrappers within tensorpack, such as sonnet/Keras/slim/tflearn/tensorlayer/....

2. Focus on large datasets.
	+ __DataFlow__ allows you to process large datasets such as ImageNet in Python without blocking the training.
	+ DataFlow has a unified interface, so you can compose and reuse them to perform complex preprocessing.

3. Focus on training speed.
	+	Tensorpack trainer is almost always faster than `feed_dict` based wrappers.
	  Even on a small CNN example, the training runs [2x faster](https://gist.github.com/ppwwyyxx/8d95da79f8d97036a7d67c2416c851b6) than the equivalent Keras code.

	+ Data-Parallel Multi-GPU training is off-the-shelf to use. For <=4 GPUs it is as fast as [tensorflow/benchmarks](https://github.com/tensorflow/benchmarks).
	  More improvements to come later.

4. Interface of extensible __Callbacks__.
	Write a callback to implement everything you want to do apart from the training iterations, and
	enable it with one line of code. Common examples include:
	+ Change hyperparameters during training
	+ Print some tensors of interest
	+ Run inference on a test dataset
	+ Run some operations once a while
	+ Send loss to your phone

## Install:

Dependencies:

+ Python 2 or 3
+ TensorFlow >= 1.0.0
+ Python bindings for OpenCV
```
pip install -U git+https://github.com/ppwwyyxx/tensorpack.git
# or add `--user` to avoid system-wide installation.
```
