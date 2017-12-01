
# tensorpack examples

Training examples with __reproducible performance__.

Reproducing a method is usually easy, but you don't know whether you've made mistakes, because wrong code will often appear to work.
Reproducible performance results are what really matters.
See [Unawareness of Deep Learning Mistakes](https://medium.com/@ppwwyyxx/unawareness-of-deep-learning-mistakes-d5b5774da0ba).


## Getting Started:
+ [An illustrative mnist example with explanation of the framework](mnist-convnet.py)
+ The same mnist example using [tf-slim](mnist-tfslim.py), [Keras layers](mnist-keras.py), [Higher-level Keras](mnist-keras-v2.py) and [with weights visualizations](mnist-visualizations.py)
+ A tiny [Cifar ConvNet](cifar-convnet.py) and [SVHN ConvNet](svhn-digit-convnet.py)
+ [A boilerplate file to start with, for your own tasks](boilerplate.py)

## Vision:
| Name | Performance |
| ---  | --- |
|	Train [ResNet](ResNet) and [ShuffleNet](ShuffleNet) on ImageNet		| reproduce paper	|
|	[Train Faster-RCNN / Mask-RCNN on COCO](FasterRCNN)				|	reproduce paper		|
| [DoReFa-Net: training binary / low-bitwidth CNN on ImageNet](DoReFa-Net) | reproduce paper |
| [Generative Adversarial Network(GAN) variants](GAN), including DCGAN, InfoGAN, <br/> Conditional GAN, WGAN, BEGAN, DiscoGAN, Image to Image, CycleGAN | visually reproduce |
| [Inception-BN and InceptionV3](Inception) | reproduce reference code |
| [Fully-convolutional Network for Holistically-Nested Edge Detection(HED)](HED) | visually reproduce |
| [Spatial Transformer Networks on MNIST addition](SpatialTransformer) | reproduce paper |
| [Visualize CNN saliency maps](Saliency) | visually reproduce |
| [Similarity learning on MNIST](SimilarityLearning) | |
| Learn steering filters with [Dynamic Filter Networks](DynamicFilterNetwork) | visually reproduce |
| Load a pre-trained [AlexNet](load-alexnet.py), [VGG16](load-vgg16.py), or [Convolutional Pose Machines](ConvolutionalPoseMachines/) | |

## Reinforcement Learning:
| Name | Performance |
| ---  | --- |
| [Deep Q-Network(DQN) variants on Atari games](DeepQNetwork), including <br/> DQN, DoubleDQN, DuelingDQN.  | reproduce paper |
| [Asynchronous Advantage Actor-Critic(A3C) on Atari games](A3C-Gym) | reproduce paper |

## Speech / NLP:
| Name | Performance |
| ---  | --- |
| [LSTM-CTC for speech recognition](CTC-TIMIT) | reproduce paper |
| [char-rnn for fun](Char-RNN) | fun |
| [LSTM language model on PennTreebank](PennTreebank) | reproduce reference code |
