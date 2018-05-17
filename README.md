![Tensorpack](.github/tensorpack.png)

Tensorpack is a training interface based on TensorFlow.

[![Build Status](https://travis-ci.org/ppwwyyxx/tensorpack.svg?branch=master)](https://travis-ci.org/ppwwyyxx/tensorpack)
[![ReadTheDoc](https://readthedocs.org/projects/tensorpack/badge/?version=latest)](http://tensorpack.readthedocs.io/en/latest/index.html)
[![Gitter chat](https://badges.gitter.im/gitterHQ/gitter.png)](https://gitter.im/tensorpack/users)
[![model-zoo](https://img.shields.io/badge/model-zoo-brightgreen.svg)](http://models.tensorpack.com)

## Features:

It's Yet Another TF high-level API, with __speed__, __readability__ and __flexibility__ built together.

1. Focus on __training speed__.
	+	Speed comes for free with tensorpack -- it uses TensorFlow in the __efficient way__ with no extra overhead.
	  On different CNNs, it runs training [1.2~5x faster](https://github.com/tensorpack/benchmarks/tree/master/other-wrappers) than the equivalent Keras code.

	+ Data-parallel multi-GPU/distributed training is off-the-shelf to use with
      one line of code. It scales as well as Google's [official benchmark](https://www.tensorflow.org/performance/benchmarks).

	+ See [tensorpack/benchmarks](https://github.com/tensorpack/benchmarks) for more benchmark scripts.

2. Focus on __large datasets__.
	+ It's unnecessary to read/preprocess data with a new language called TF.
		Tensorpack helps you load large datasets (e.g. ImageNet) in __pure Python__ with autoparallelization.

3. It's not a model wrapper.
	+ There are too many symbolic function wrappers in the world. Tensorpack includes only a few common models.
	  But you can use any symbolic function library inside tensorpack, including tf.layers/Keras/slim/tflearn/tensorlayer/....

See [tutorials](http://tensorpack.readthedocs.io/tutorial/index.html#user-tutorials) to know more about these features.

## [Examples](examples):

We refuse toy examples.
Instead of showing you 10 arbitrary networks trained on toy datasets,
[tensorpack examples](examples) faithfully replicate papers and care about reproducing numbers,
demonstrating its flexibility for actual research.

### Vision:
+ [Train ResNet](examples/ResNet) and [other models](examples/ImageNetModels) on ImageNet.
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

## Install:

Dependencies:

+ Python 2.7 or 3
+ Python bindings for OpenCV (Optional, but required by a lot of features)
+ TensorFlow >= 1.3.0 (Optional if you only want to use `tensorpack.dataflow` alone as a data processing library)
```
# install git, then:
pip install -U git+https://github.com/ppwwyyxx/tensorpack.git
# or add `--user` to avoid system-wide installation.
```

## Citing Tensorpack:

If you use Tensorpack in your research or wish to refer to the examples, please cite with:
```
@misc{wu2016tensorpack,
  title={Tensorpack},
  author={Wu, Yuxin and others},
  howpublished={\url{https://github.com/tensorpack/}},
  year={2016}
}
```
