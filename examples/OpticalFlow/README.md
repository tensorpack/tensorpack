## OpticalFlow - FlowNet2

Reproduces
[FlowNet 2.0: Evolution of Optical Flow Estimation with Deep Networks](https://arxiv.org/abs/1612.01925)
by Ilg et al.

Given two images, the network is trained to predict the optical flow between these images.

<p align="center"> <img src="./preview.jpg" width="100%"> </p>

* Top: both input images from Flying Chairs, ground-truth, Caffe output
* Bottom: FlowNet2-C, FlowNet2-S, FlowNet2 results (when converted to TensorFlow)

The authors report the AEE of *2.03* (Caffe Model) on Sintel-clean and our implementation gives an AEE of *2.10*, which is better than other TensorFlow implementations.


### Usage

1. Download the pre-trained model:

```bash
wget http://models.tensorpack.com/opticalflow/flownet2-s.npz
wget http://models.tensorpack.com/opticalflow/flownet2-c.npz
wget http://models.tensorpack.com/opticalflow/flownet2.npz

```

*Note:* Using these weights, requires to accept the author's license:

```
Pre-trained weights are provided for research purposes only and without any warranty.
Any commercial use of the pre-trained weights requires FlowNet2 authors consent.
```

2. Run inference

```bash
python python flownet2.py --gpu 0 \
        --left left_img.ppm \
        --right right_img.ppm \
        --load flownet2-s.npz --model "flownet2-s"
python python flownet2.py --gpu 0 \
        --left left_img.ppm \
        --right right_img.ppm \
        --load flownet2-c.npz --model "flownet2-c"
python python flownet2.py --gpu 0 \
        --left left_img.ppm \
        --right right_img.ppm \
        --load flownet2.npz --model "flownet2"
```

*Note:* The current `correlation`-layer implementation is pure TensorFlow code and has a high memory consumption and might be slow. See the details in [`flownet_models.py`](./flownet_models.py) for a faster version with a smaller memory footprint.