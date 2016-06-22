This is the official script to load and run pretrained model for the paper:

[DoReFa-Net: Training Low Bitwidth Convolutional Neural Networks with Low Bitwidth Gradients](http://arxiv.org/abs/1606.06160), by Zhou et al.

The provided model is an AlexNet with 1 bit weights, 2 bit activations, trained with 4 bit gradients.

## Preparation:

To use the script. You'll need:

+ [TensorFlow](https://tensorflow.org) >= 0.8

+ OpenCV bindings for Python

+ [tensorpack](https://github.com/ppwwyyxx/tensorpack):

```
git clone https://github.com/ppwwyyxx/tensorpack
pip install --user -r tensorpack/requirements.txt
export PYTHONPATH=$PYTHONPATH:`readlink -f tensorpack`
```

+ Download the model at [google drive](https://drive.google.com/open?id=0B308TeQzmFDLa0xOeVQwcXg1ZjQ)

## Load and run the model
We published the model in two file formats:

+ `alexnet.npy`. It's simply a dict of {param name: value}.
You can load it with `np.load('alexnet.npy').item()` for other purposes.
Run the model with:

```
./alexnet.py --load alexnet.npy [--input img.jpg] [--data path/to/data]
```

+ `alexnet.meta` + `alexnet.tfmodel`. A TensorFlow MetaGraph proto and a saved checkpoint.

```
./alexnet.py --graph alexnet.meta --load alexnet.tfmodel [--input path/to/img.jpg] [--data path/to/ILSVRC12]
```

In both cases, one of `--data` or `--input` must be present, to either run classification on some input images, or run evaluation on ILSVRC12 validation set.
To eval on ILSVRC12, `path/to/ILSVRC12` must have a subdirectory named 'val' containing all the validation images.

## Support

Please use [github issues](https://github.com/ppwwyyxx/tensorpack/issues) for any issues related to the code.
Send email to the authors for other questions related to the paper.

## Citation

If you use our models in your research, please cite:
```
@article{zhou2016dorefa,
  author    = {Shuchang Zhou and Zekun Ni and Xinyu Zhou and He Wen and Yuxin Wu and Yuheng Zou},
  title     = {DoReFa-Net: Training Low Bitwidth Convolutional Neural Networks with Low Bitwidth Gradients},
  journal   = {CoRR},
  volume    = {abs/1606.06160},
  year      = {2016},
  url       = {http://arxiv.org/abs/1606.06160},
}
```
