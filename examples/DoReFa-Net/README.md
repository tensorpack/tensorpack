Code and model for the paper:

[DoReFa-Net: Training Low Bitwidth Convolutional Neural Networks with Low Bitwidth Gradients](http://arxiv.org/abs/1606.06160), by Zhou et al.

We hosted a demo at CVPR16 on behalf of Megvii, Inc, running a real-time 1/4-VGG size DoReFa-Net on ARM and half-VGG size DoReFa-Net on FPGA.
We're not planning to release those runtime bit-op libraries for now. In this repo, bit operations are run in float32.

Pretrained model for 1-2-6-AlexNet is available at
[google drive](https://drive.google.com/a/%20megvii.com/folderview?id=0B308TeQzmFDLa0xOeVQwcXg1ZjQ).
It's provided in the format of numpy dictionary, so it should be very easy to port into other applications.

## Preparation:

To use the script. You'll need:

+ [TensorFlow](https://tensorflow.org) >= 0.10

+ OpenCV bindings for Python

+ [tensorpack](https://github.com/ppwwyyxx/tensorpack):

```
git clone https://github.com/ppwwyyxx/tensorpack
pip install --user -r tensorpack/requirements.txt
pip install --user pyzmq scipy
export PYTHONPATH=$PYTHONPATH:`readlink -f tensorpack`
```

+ Look at the docstring in `svhn-digit-dorefa.py` or `alexnet-dorefa.py` to see detailed usage and performance.

## Support

Please use [github issues](https://github.com/ppwwyyxx/tensorpack/issues) for any issues related to the code itself.
Send email to the authors for general questions related to the paper.

## Citation

If you use our code or models in your research, please cite:
```
@article{zhou2016dorefa,
  author    = {Shuchang Zhou and Yuxin Wu and Zekun Ni and Xinyu Zhou and He Wen and Yuheng Zou},
  title     = {DoReFa-Net: Training Low Bitwidth Convolutional Neural Networks with Low Bitwidth Gradients},
  journal   = {CoRR},
  volume    = {abs/1606.06160},
  year      = {2016},
  url       = {http://arxiv.org/abs/1606.06160},
}
```
