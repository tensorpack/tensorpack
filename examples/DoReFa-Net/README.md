Code and model for the paper:

[DoReFa-Net: Training Low Bitwidth Convolutional Neural Networks with Low Bitwidth Gradients](http://arxiv.org/abs/1606.06160), by Zhou et al.

It also contains an implementation of the following papers:
+ [Binary Weight Network](https://arxiv.org/abs/1511.00363), with (W,A,G)=(1,32,32).
+ [Trained Ternary Quantization](https://arxiv.org/abs/1612.01064), with (W,A,G)=(t,32,32).
+ [Binarized Neural Networks](https://arxiv.org/abs/1602.02830), with (W,A,G)=(1,1,32).

These quantization techniques achieves the following ImageNet performance in this implementation:

| Model              | W,A,G       | Top 1 Error |
|:-------------------|-------------|------------:|
| Full Precision     | 32,32,32    |      41.4%  |
| TTQ                | t,32,32     |      41.9%  |
| BWN                | 1,32,32     |      44.3%  |
| BNN                | 1,1,32      |      53.4%  |
| DoReFa             | 1,2,6       |      47.6%  |
| DoReFa             | 1,2,4       |      58.4%  |

We hosted a demo at CVPR16 on behalf of Megvii, Inc, running a real-time 1/4-VGG size DoReFa-Net on ARM and half-VGG size DoReFa-Net on FPGA.
We're not planning to release our C++ runtime for bit-operations.
In this repo, bit operations are performed through `tf.float32`.

Pretrained model for (1,4,32)-ResNet18 and (1,2,6)-AlexNet are available at
[tensorpack model zoo](http://models.tensorpack.com/DoReFa-Net/).
They're provided in the format of numpy dictionary, so it should be very easy to port into other applications.
The __binary-weight 4-bit-activation ResNet-18__ model has 59.2% top-1 validation accuracy.

Alternative link to this page: [http://dorefa.net](http://dorefa.net)

## Preparation:

+ Install [tensorpack](https://github.com/ppwwyyxx/tensorpack) and scipy.

+ Look at the docstring in `*-dorefa.py` to see detailed usage and performance.

## Support

Please use [github issues](https://github.com/ppwwyyxx/tensorpack/issues) for any issues related to the code itself.
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
