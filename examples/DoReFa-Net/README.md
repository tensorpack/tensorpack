Official code and model for the paper:

+ [DoReFa-Net: Training Low Bitwidth Convolutional Neural Networks with Low Bitwidth Gradients](http://arxiv.org/abs/1606.06160).

It also contains an implementation of the following papers:
+ [Binary Weight Network](https://arxiv.org/abs/1511.00363), with (W,A,G)=(1,32,32).
+ [Trained Ternary Quantization](https://arxiv.org/abs/1612.01064), with (W,A,G)=(t,32,32).
+ [Binarized Neural Networks](https://arxiv.org/abs/1602.02830), with (W,A,G)=(1,1,32).

Alternative link to this page: [http://dorefa.net](http://dorefa.net)

## Results:
This is a good set of baselines for research in model quantization.
These quantization techniques, when applied on AlexNet, achieves the following ImageNet performance in this implementation:

| Model                              | Bit Width <br/> (weights, activations, gradients) | Top 1 Validation Error <sup>[1](#ft1)</sup>                                      |
|:----------------------------------:|:-------------------------------------------------:|:--------------------------------------------------------------------------------:|
| Full Precision<sup>[2](#ft2)</sup> | 32,32,32                                          | 40.3%                                                                            |
| TTQ                                | t,32,32                                           | 42.0%                                                                            |
| BWN                                | 1,32,32                                           | 44.3% [:arrow_down:](http://models.tensorpack.com/DoReFa-Net/AlexNet-1,32,32.npz) |
| BNN                                | 1,1,32                                            | 51.5% [:arrow_down:](http://models.tensorpack.com/DoReFa-Net/AlexNet-1,1,32.npz) |
| DoReFa                             | 8,8,8                                             | 42.0% [:arrow_down:](http://models.tensorpack.com/DoReFa-Net/AlexNet-8,8,8.npz)  |
| DoReFa                             | 1,2,32                                            | 46.6%                                                                            |
| DoReFa                             | 1,2,6                                             | 46.8% [:arrow_down:](http://models.tensorpack.com/DoReFa-Net/AlexNet-1,2,6.npz)  |
| DoReFa                             | 1,2,4                                             | 54.0%                                                                            |

 <a id="ft1">1</a>: These numbers were obtained by training on 8 GPUs with a total batch size of 256 (otherwise the performance may become slightly different).
The DoReFa-Net models reach slightly better performance than our paper, due to
more sophisticated augmentations.

 <a id="ft2">2</a>: Not directly comparable with the original AlexNet. Check out
 [../ImageNetModels](../ImageNetModels) for a more faithful implementation of the original AlexNet.

## Speed:
__DoReFa-Net works on mobile and FPGA!__
We hosted a demo at CVPR16 on behalf of Megvii, Inc, running a 1/4-VGG size DoReFa-Net on a phone and a half-VGG size DoReFa-Net on an FPGA, in real time.
DoReFa-Net and its variants have been deployed widely in Megvii's embeded products.

This code release is meant for research purpose. We're not planning to release our C++ runtime for bit-operations.
In this implementation, quantized operations are all performed through `tf.float32`. They don't make your network faster.

## Use

+ Install TensorFlow ≥ 1.7, tensorpack and scipy.

+ Look at the docstring in `*-dorefa.py` to see detailed usage and performance.

Pretrained model for (1,4,32)-ResNet18 and (1,2,6)-AlexNet are available at
[tensorpack model zoo](http://models.tensorpack.com/DoReFa-Net/).
They're provided in the format of numpy dictionary.
The __binary-weight 4-bit-activation ResNet-18__ model has 59.2% top-1 validation accuracy.


## Support

Please use [github issues](https://github.com/tensorpack/tensorpack/issues) for any issues related to the code itself.
Please send email to the authors for general questions related to the paper.

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
