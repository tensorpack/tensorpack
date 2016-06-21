This is the official script to load and run pretrained model for the paper:

[DoReFa-Net: Training Low Bitwidth Convolutional Neural Networks with Low Bitwidth Gradients](http://arxiv.org/abs/1606.06160), by Zhou et al.

(Work in Progress. More instructions to come soon)

This is an AlexNet model with 1 bit weights, 2 bit activations, trained with 4 bit gradients.

## Preparation:

To use the script. You'll need:

+ [TensorFlow](tensorflow.org) >= 0.8

+ [tensorpack](https://github.com/ppwwyyxx/tensorpack) and pyzmq:

```
git clone https://github.com/ppwwyyxx/tensorpack
pip install --user -r tensorpack/requirements.txt
pip install --user pyzmq
export PYTHONPATH=$PYTHONPATH:`readlink -f tensorpack`
```

## Load and run the model
We provide two format for the model:

1. alexnet.npy. It's simply a numpy dict of {param name: value}. To load:

```
./alexnet.py --load alexnet.npy [--input img.jpg] [--data path/to/data]
```

2. alexnet.meta + alexnet.tfmodel. A TensorFlow MetaGraph proto and a saved checkpoint.

```
./alexnet.py --graph alexnet.meta --load alexnet.tfmodel [--input path/to/img.jpg] [--data path/to/ILSVRC12]
```

One of `--data` or `--input` must be present, to either run classification on some input images, or run evaluation on ILSVRC12 validation set.
To eval on ILSVRC12, `path/to/ILSVRC12` must have a subdirectory named 'val' containing all the validation images.
