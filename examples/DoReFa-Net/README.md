Official code and model for the paper:

+ [DoReFa-Net: Training Low Bitwidth Convolutional Neural Networks with Low Bitwidth Gradients](http://arxiv.org/abs/1606.06160).

It also contains an implementation of the following papers:
+ [Binary Weight Network](https://arxiv.org/abs/1511.00363), with (W,A,G)=(1,32,32).
+ [Trained Ternary Quantization](https://arxiv.org/abs/1612.01064), with (W,A,G)=(t,32,32).
+ [Binarized Neural Networks](https://arxiv.org/abs/1602.02830), with (W,A,G)=(1,1,32).

This is a good set of baselines for research in model quantization.
These quantization techniques achieves the following ImageNet performance in this implementation:

| Model          | W,A,G    | Top 1 Validation Error |
|:---------------|----------|-----------------------:|
| Full Precision | 32,32,32 |                  40.3% |
| TTQ            | t,32,32  |                  42.0% |
| BWN            | 1,32,32  |                  44.6% |
| BNN            | 1,1,32   |                  51.9% |
| DoReFa         | 1,2,32   |                  46.6% |
| DoReFa         | 1,2,6    |                  46.8% |
| DoReFa         | 1,2,4    |                  54.0% |

These numbers were obtained by training on 8 GPUs with a total batch size of 256.
The DoReFa-Net models reach slightly better performance than our paper, due to
more sophisticated augmentations.

We hosted a demo at CVPR16 on behalf of Megvii, Inc, running a real-time 1/4-VGG size DoReFa-Net on ARM and half-VGG size DoReFa-Net on FPGA.
We're not planning to release our C++ runtime for bit-operations.
In this repo, quantized operations are all performed through `tf.float32`.

Pretrained model for (1,4,32)-ResNet18 and (1,2,6)-AlexNet are available at
[tensorpack model zoo](http://models.tensorpack.com/DoReFa-Net/).
They're provided in the format of numpy dictionary.
The __binary-weight 4-bit-activation ResNet-18__ model has 59.2% top-1 validation accuracy.

Alternative link to this page: [http://dorefa.net](http://dorefa.net)

## Use

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
