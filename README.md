![Tensorpack](.github/tensorpack.png)

Tensorpack is a training interface based on TensorFlow.

[![Build Status](https://travis-ci.org/ppwwyyxx/tensorpack.svg?branch=master)](https://travis-ci.org/ppwwyyxx/tensorpack)
[![ReadTheDoc](https://readthedocs.org/projects/tensorpack/badge/?version=latest)](http://tensorpack.readthedocs.io/en/latest/index.html)
[![Gitter chat](https://badges.gitter.im/gitterHQ/gitter.png)](https://gitter.im/tensorpack/users)

See some [examples](examples) to learn about the framework. Everything runs on multiple GPUs, because why not?

### Vision:
+ [Train ResNet/SE-ResNet on ImageNet](examples/ResNet)
+ [Train Faster-RCNN / Mask-RCNN on COCO object detection](examples/FasterRCNN)
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

1. Focus on __training speed__.
	+	Speed comes for free with tensorpack -- it uses TensorFlow in the __correct way__ with no extra overhead.
	  On various CNNs, it runs 1.5~1.7x faster than the equivalent Keras code.

	+ Data-parallel multi-GPU training is off-the-shelf to use. It is as fast as Google's [official benchmark](https://www.tensorflow.org/performance/benchmarks).
		You cannot beat its speed unless you're a TensorFlow expert.

	+ See [tensorpack/benchmarks](https://github.com/tensorpack/benchmarks) for some benchmark scripts.

2. Focus on __large datasets__.
	+ It's painful to read/preprocess data through TF.
		Tensorpack helps you load large datasets (e.g. ImageNet) in __pure Python__ with autoparallelization.

3. It's not a model wrapper.
	+ There are already too many symbolic function wrappers.
		Tensorpack includes only a few common models,
	  but you can use any other wrappers within tensorpack, including sonnet/Keras/slim/tflearn/tensorlayer/....

See [tutorials](http://tensorpack.readthedocs.io/en/latest/tutorial/index.html) to know more about these features.

## Install:

Dependencies:

+ Python 2.7 or 3
+ Python bindings for OpenCV (Optional, but required by a lot of features)
+ TensorFlow >= 1.2.0 (Optional if you only want to use `tensorpack.dataflow` alone as a data processing library)
```
# install git, then:
pip install -U git+https://github.com/ppwwyyxx/tensorpack.git
# or add `--user` to avoid system-wide installation.
```
