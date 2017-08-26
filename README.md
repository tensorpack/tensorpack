# tensorpack
A neural net training interface based on TensorFlow.

[![Build Status](https://travis-ci.org/ppwwyyxx/tensorpack.svg?branch=master)](https://travis-ci.org/ppwwyyxx/tensorpack)
[![badge](https://readthedocs.org/projects/pip/badge/?version=latest)](http://tensorpack.readthedocs.io/en/latest/index.html)

See some [examples](examples) to learn about the framework:

### Vision:
+ [Multi-GPU training of ResNet on ImageNet](examples/ResNet)
+ [Generative Adversarial Network(GAN) variants](examples/GAN), including DCGAN, InfoGAN, Conditional GAN, WGAN, BEGAN, DiscoGAN, Image to Image, CycleGAN.
+ [DoReFa-Net: train binary / low-bitwidth CNN on ImageNet](examples/DoReFa-Net)
+ [Fully-convolutional Network for Holistically-Nested Edge Detection(HED)](examples/HED)
+ [Spatial Transformer Networks on MNIST addition](examples/SpatialTransformer)
+ [Visualize CNN saliency maps](examples/Saliency)
+ [Similarity learning on MNIST](examples/SimilarityLearning)

### Reinforcement Learning:
+ [Deep Q-Network(DQN) variants on Atari games](examples/DeepQNetwork), including DQN, DoubleDQN, DuelingDQN.
+ [Asynchronous Advantage Actor-Critic(A3C) with demos on OpenAI Gym](examples/A3C-Gym)

### Speech / NLP:
+ [LSTM-CTC for speech recognition](examples/CTC-TIMIT)
+ [char-rnn for fun](examples/Char-RNN)
+ [LSTM language model on PennTreebank](examples/PennTreebank)

Examples are not only for demonstration of the framework -- you can train them and reproduce the results in papers.

## Features:

It's Yet Another TF wrapper, but different in:
1. Not focus on models.
	+ There are already too many symbolic function wrappers.
		Tensorpack includes only a few common models,
	  but you can use any other wrappers within tensorpack, such as sonnet/Keras/slim/tflearn/tensorlayer/....

2. Focus on __training speed__.
	+	Speed comes for free with tensorpack -- it uses TensorFlow in the correct way.
	  Even on a tiny CNN example, the training runs [1.6x faster](https://gist.github.com/ppwwyyxx/8d95da79f8d97036a7d67c2416c851b6) than the equivalent Keras code.

	+ Data-parallel multi-GPU training is off-the-shelf to use. It is as fast as Google's [official benchmark](https://www.tensorflow.org/performance/benchmarks).

	+ Data-parallel distributed training is off-the-shelf to use. It is as slow as Google's official benchmark.

3. Focus on __large datasets__.
	+ It's painful to read/preprocess data through TF. tensorpack helps you load large datasets (e.g. ImageNet) in __pure Python__ with autoparallelization.

4. Interface of extensible __Callbacks__.
	Write a callback to implement everything you want to do apart from the training iterations, and
	enable it with one line of code. Common examples include:
	+ Change hyperparameters during training
	+ Print some tensors of interest
	+ Monitor GPU utilization
	+ Send error rate to your phone

See [tutorials](http://tensorpack.readthedocs.io/en/latest/tutorial/index.html) to know more about these features.

## Install:

Dependencies:

+ Python 2 or 3
+ TensorFlow >= 1.0.0 (>=1.1.0 for Multi-GPU)
+ Python bindings for OpenCV (Optional, but required by a lot of features)
```
pip install -U git+https://github.com/ppwwyyxx/tensorpack.git
# or add `--user` to avoid system-wide installation.
```
Besides, if you only want to use `tensorpack.dataflow` alone as a data processing library, TensorFlow is also optional.
