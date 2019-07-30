![Tensorpack](.github/tensorpack.png)

Tensorpack is a neural network training interface based on TensorFlow. 
This specifc branch is for 
1. Inference in CPU 
2. NHWC format support
3. Save forzen graph protobuf format support.
4. Run Inference from frozen graph support.
5. Run Inference on single image support
6. Run Inference on dataset support
7. Run Inference on dummy data support

Please scroll to the end of this page for CPU related directions.

[![Build Status](https://travis-ci.org/tensorpack/tensorpack.svg?branch=master)](https://travis-ci.org/tensorpack/tensorpack)
[![ReadTheDoc](https://readthedocs.org/projects/tensorpack/badge/?version=latest)](http://tensorpack.readthedocs.io)
[![Gitter chat](https://img.shields.io/badge/chat-on%20gitter-46bc99.svg)](https://gitter.im/tensorpack/users)
[![model-zoo](https://img.shields.io/badge/model-zoo-brightgreen.svg)](http://models.tensorpack.com)
## Features:

It's Yet Another TF high-level API, with __speed__, and __flexibility__ built together.

1. Focus on __training speed__.
	+ Speed comes for free with Tensorpack -- it uses TensorFlow in the __efficient way__ with no extra overhead.
	  On common CNNs, it runs training [1.2~5x faster](https://github.com/tensorpack/benchmarks/tree/master/other-wrappers) than the equivalent Keras code.
		Your training can probably gets faster if written with Tensorpack.

	+ Data-parallel multi-GPU/distributed training strategy is off-the-shelf to use.
    It scales as well as Google's [official benchmark](https://www.tensorflow.org/performance/benchmarks).

	+ See [tensorpack/benchmarks](https://github.com/tensorpack/benchmarks) for
    some benchmark scripts.

2. Focus on __large datasets__.
	+ [You don't usually need `tf.data`](http://tensorpack.readthedocs.io/tutorial/extend/input-source.html#tensorflow-reader-cons).
    Symbolic programming often makes data processing harder.
	  Tensorpack helps you efficiently process large datasets (e.g. ImageNet) in __pure Python__ with autoparallelization.

3. It's not a model wrapper.
	+ There are too many symbolic function wrappers in the world. Tensorpack includes only a few common models.
	  But you can use any symbolic function library inside Tensorpack, including tf.layers/Keras/slim/tflearn/tensorlayer/....

See [tutorials and documentations](http://tensorpack.readthedocs.io/tutorial/index.html#user-tutorials) to know more about these features.

## Examples:

We refuse toy examples. We refuse low-quality implementations.
Unlike most open source repos which only __implement__ papers,
[Tensorpack examples](examples) faithfully __reproduce__ papers,
demonstrating its __flexibility__ for actual research.

### Vision:
+ [Train ResNet](examples/ResNet) and [other models](examples/ImageNetModels) on ImageNet.
+ [Train Mask/Faster R-CNN on COCO object detection](examples/FasterRCNN)
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

+ Python 2.7 or 3.3+. Python 2.7 is supported until [it retires in 2020](https://pythonclock.org/).
+ Python bindings for OpenCV. (Optional, but required by a lot of features)
+ TensorFlow ≥ 1.3, < 2. (Optional, if you only want to use `tensorpack.dataflow` alone as a data processing library)
```
pip install --upgrade git+https://github.com/tensorpack/tensorpack.git
# or add `--user` to install to user's local directories
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

## CPU NHWC FORMAT INSTRUCTIONS
Conda Environment and Python version tested.
```
conda create -n fasterrcnnfpn python=3.6
conda activate fasterrcnnfpn
```

Install Dependecies
```
pip install numpy
pip install keras_preprocessing
pip install tqdm
pip install tensorpack
pip install cython
pip install pycocotools
pip install matplotlib
```

## Load from chekpoint and save protobuf file
```
python train.py --evaluate ./temp/output.json --load ./temp/model.ckpt  --savepb --config MODE_MASK=False MODE_FPN=True DATA.BASEDIR=./temp/ DATA.VAL='val2017' TRAIN.NUM_GPUS=1
```
## Run inference 
on single image
```
python train.py --predict ./temp/val2017/000000000785.jpg --load ./temp/model.ckpt --loadfrozenpb --model_name=Fasterrcnnfpn_freezed_quantized_int8_s2.pb --config MODE_MASK=False MODE_FPN=True DATA.BASEDIR=./temp/ DATA.VAL='val2017' TRAIN.NUM_GPUS=1
```
on dataset(Accuracy run)
```
python train.py --evaluate ./temp/output.json --load ./temp/model.ckpt --loadfrozenpb --model_name=/Fasterrcnnfpn_freezed_quantized_int8_final.pb --config MODE_MASK=False MODE_FPN=True DATA.BASEDIR=./temp/ DATA.VAL='val2017' TRAIN.NUM_GPUS=1
 
python train.py --predict ./temp/val2017/000000000785.jpg --load ./temp/model.ckpt --loadfrozenpb --model_name=Fasterrcnnfpn_freezed_quantized_int8_final.pb --config MODE_MASK=False MODE_FPN=True DATA.BASEDIR=./temp/ DATA.VAL='val2017' TRAIN.NUM_GPUS=1

```
on dummy data(performance run)
```
python run_faster_rcnn_fpn_inferencegraph_dummydata.py --model_name=Fasterrcnnfpn_freezed_optimized_fp32.pb --image_count=100
python run_faster_rcnn_fpn_inferencegraph_dummydata.py --model_name=Fasterrcnnfpn_freezed_quantized_int8_final.pb --image_count=100

```
