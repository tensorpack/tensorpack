
# tensorpack examples

Training examples with __reproducible__ and meaningful performance.

## Vision:
+ [An illustrative mnist example with explanation of the framework](mnist-convnet.py)
+ [A tiny SVHN ConvNet with 97.8% accuracy](svhn-digit-convnet.py)
+ [DoReFa-Net: training binary / low-bitwidth CNN on ImageNet](DoReFa-Net)
+ [ResNet for ImageNet/Cifar10/SVHN](ResNet)
+ [Inception-BN with 71% accuracy](Inception/inception-bn.py)
+ [InceptionV3 with 74% accuracy (similar to the official code)](Inception/inceptionv3.py)
+ [Fully-convolutional Network for Holistically-Nested Edge Detection(HED)](HED)
+ [Spatial Transformer Networks on MNIST addition](SpatialTransformer)
+ Load a pre-trained [AlexNet](load-alexnet.py) or [VGG16](load-vgg16.py) model.
+ Load a pre-trained [Convolutional Pose Machines](ConvolutionalPoseMachines/).

## Reinforcement Learning:
+ [Deep Q-Network(DQN) variants on Atari games](Atari2600)
+ [Asynchronous Advantage Actor-Critic(A3C) with demos on OpenAI Gym](OpenAIGym)

## Unsupervised:
+ [Generative Adversarial Network(GAN) variants, including DCGAN, Image2Image, InfoGAN](GAN)

## Speech / NLP:
+ [LSTM-CTC for speech recognition](TIMIT)
+ [char-rnn for fun](char-rnn)


Note to contributors:

We have a high bar for examples. It needs to satisfy one of the following:
+ Reproduce performance of a published or well-known paper.
+ Get state-of-the-art performance on some task.
+ Illustrate a new way of using the library that are currently not covered.
