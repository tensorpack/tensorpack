
Example code to convert, load and run inference of some Caffe models.
Require caffe python bindings to be installed.
Converted models can also be found at [tensorpack model zoo](http://models.tensorpack.com).

## AlexNet:

Download: https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet

Convert:
```
python -m tensorpack.utils.loadcaffe PATH/TO/CAFFE/{deploy.prototxt,bvlc_alexnet.caffemodel} alexnet.npz
```

Run: `./load-alexnet.py --load alexnet.npz --input cat.png`

## VGG

Download: https://gist.github.com/ksimonyan/211839e770f7b538e2d8

Convert:
```
python -m tensorpack.utils.loadcaffe \
            PATH/TO/VGG/{VGG_ILSVRC_16_layers_deploy.prototxt,VGG_ILSVRC_16_layers.caffemodel} vgg16.npz
```

Run: `./load-vgg16.py --load vgg16.npz --input cat.png`


## ResNet

To load caffe version of ResNet, see instructions in [ResNet examples](../ResNet).

## Convolutional Pose Machines

Download:
```
wget http://pearl.vasc.ri.cmu.edu/caffe_model_github/model/_trained_MPI/pose_iter_320000.caffemodel
wget https://github.com/shihenw/convolutional-pose-machines-release/raw/master/model/_trained_MPI/pose_deploy_resize.prototxt
```

Convert:
```
python -m tensorpack.utils.loadcaffe pose_deploy_resize.prototxt pose_iter_320000.caffemodel CPM-original.npz
```

Run: `python load-cpm.py --load CPM-original.npz --input test.jpg`

Input image will get resized to 368x368. Note that this CPM comes __without__ person detection, so the
person has to be in the center of the image (and not too small).

![demo](demo-cpm.jpg)

Also check out [Stereo Pose Machines](https://github.com/ppwwyyxx/Stereo-Pose-Machines), a real-time CPM application based on tensorpack.
