
# tensorpack examples

Training examples with __reproducible__ and meaningful performance.

## Getting Started:
+ [An illustrative mnist example with explanation of the framework](mnist-convnet.py)
+ The same mnist example using [tf-slim](mnist-tfslim.py), [Keras](mnist-keras.py), and [with weights visualizations](mnist-visualizations.py)
+ [A boilerplate file to start with, for your own tasks](boilerplate.py)

## Vision:
+ [A tiny SVHN ConvNet with 97.8% accuracy](svhn-digit-convnet.py)
+ [Multi-GPU training of ResNet on ImageNet](ResNet)
+ [DoReFa-Net: training binary / low-bitwidth CNN on ImageNet](DoReFa-Net)
+ [Generative Adversarial Network(GAN) variants](GAN), including DCGAN, InfoGAN, Conditional GAN, WGAN, BEGAN, DiscoGAN, Image to Image, CycleGAN.
+ [Inception-BN and InceptionV3](Inception)
+ [Fully-convolutional Network for Holistically-Nested Edge Detection(HED)](HED)
+ [Spatial Transformer Networks on MNIST addition](SpatialTransformer)
+ [Visualize CNN saliency maps](Saliency)
+ [Similarity learning on MNIST](SimilarityLearning)
+ Load a pre-trained [AlexNet](load-alexnet.py) or [VGG16](load-vgg16.py) model.
+ Load a pre-trained [Convolutional Pose Machines](ConvolutionalPoseMachines/).

## Reinforcement Learning:
+ [Deep Q-Network(DQN) variants on Atari games](DeepQNetwork), including DQN, DoubleDQN, DuelingDQN.
+ [Asynchronous Advantage Actor-Critic(A3C) with demos on OpenAI Gym](A3C-Gym)

## Speech / NLP:
+ [LSTM-CTC for speech recognition](CTC-TIMIT)
+ [char-rnn for fun](Char-RNN)
+ [LSTM language model on PennTreebank](PennTreebank)


Note to contributors:

Example needs to satisfy one of the following:
+ Reproduce performance of a published or well-known paper.
+ Get state-of-the-art performance on some task.
+ Illustrate a new way of using the library that is currently not covered.
