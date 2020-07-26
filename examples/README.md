
# Tensorpack Examples

Training examples with __reproducible performance__.

__The word "reproduce" should always mean reproduce performance__.
With the magic of SGD, wrong deep learning code often appears to work, especially if you try it on toy datasets.
Github is full of deep learning code that "implements" but does not "reproduce"
methods, and you'll not know whether the implementation is actually correct.
See [Unawareness of Deep Learning Mistakes](https://medium.com/@ppwwyyxx/unawareness-of-deep-learning-mistakes-d5b5774da0ba).

We refuse toy examples.
Instead of showing tiny CNNs trained on MNIST/Cifar10,
we provide training scripts that reproduce well-known papers.

We refuse low-quality implementations.
Unlike most open source repos which only __implement__ methods,
[Tensorpack examples](examples) faithfully __reproduce__
experiments and performance in the paper,
so you're confident that they are correct.


## Getting Started:
These are the only toy examples in tensorpack. They are supposed to be just demos.
+ [An illustrative MNIST example with explanation of the framework](basics/mnist-convnet.py)
+ Tensorpack supports any symbolic libraries. See the same MNIST example written with [tf.layers](basics/mnist-tflayers.py), and [with weights visualizations](basics/mnist-visualizations.py)
+ A tiny [Cifar ConvNet](basics/cifar-convnet.py) and [SVHN ConvNet](basics/svhn-digit-convnet.py)
+ If you've used Keras, check out [Keras+Tensorpack examples](keras)
+ [A boilerplate file to start with, for your own tasks](boilerplate.py)

## Vision:
| Name                                                                                                                                                  | Performance                 |
| ---                                                                                                                                                   | ---                         |
| Train [ResNet](ResNet), [ShuffleNet and other models](ImageNetModels) on ImageNet                                                                     | reproduce 10 papers         |
| [Train Mask/Faster R-CNN on COCO](FasterRCNN)                                                                                                         | reproduce 7 papers          |
| [Unsupervised learning with Momentum Contrast](https://github.com/ppwwyyxx/moco.tensorflow) (MoCo)                                                    | reproduce 2 papers          |
| [Generative Adversarial Network(GAN) variants](GAN), including DCGAN, InfoGAN, <br/> Conditional GAN, WGAN, BEGAN, DiscoGAN, Image to Image, CycleGAN | visually reproduce 8 papers |
| [DoReFa-Net: training binary / low-bitwidth CNN on ImageNet](DoReFa-Net)                                                                              | reproduce 4 papers          |
| [Adversarial training with state-of-the-art robustness](https://github.com/facebookresearch/ImageNet-Adversarial-Training)                            | official code for the paper |
| [Fully-convolutional Network for Holistically-Nested Edge Detection(HED)](HED)                                                                        | visually reproduce          |
| [Spatial Transformer Networks on MNIST addition](SpatialTransformer)                                                                                  | reproduce the paper         |
| [Visualize CNN saliency maps](Saliency)                                                                                                               | visually reproduce          |
| [Similarity learning on MNIST](SimilarityLearning)                                                                                                    |                             |
| Single-image super-resolution using [EnhanceNet](SuperResolution)                                                                                     |                             |
| Learn steering filters with [Dynamic Filter Networks](DynamicFilterNetwork)                                                                           | visually reproduce          |
| Load a pre-trained [AlexNet, VGG, or Convolutional Pose Machines](CaffeModels)                                                                        |                             |
| Load a pre-trained [FlowNet2-S, FlowNet2-C, FlowNet2](OpticalFlow)                                                                                    |                             |

## Reinforcement Learning:
| Name                                                                                                     | Performance         |
| ---                                                                                                      | ---                 |
| [Deep Q-Network(DQN) variants on Atari games](DeepQNetwork), including <br/> DQN, DoubleDQN, DuelingDQN. | reproduce the paper |
| [Asynchronous Advantage Actor-Critic(A3C) on Atari games](A3C-Gym)                                       | reproduce the paper |

## Speech / NLP:
| Name                                                | Performance              |
| ---                                                 | ---                      |
| [LSTM-CTC for speech recognition](CTC-TIMIT)        | reproduce the paper      |
| [char-rnn for fun](Char-RNN)                        | fun                      |
| [LSTM language model on PennTreebank](PennTreebank) | reproduce reference code |
