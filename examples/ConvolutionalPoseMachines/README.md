# Convolutional Pose Machines

A script to load and run pretrained CPM model released by Shih-En. The original code in caffe is [here](https://github.com/shihenw/convolutional-pose-machines-release).

## Usage:
```
# download the relased caffe model:
wget http://pearl.vasc.ri.cmu.edu/caffe_model_github/model/_trained_MPI/pose_iter_320000.caffemodel
wget https://github.com/shihenw/convolutional-pose-machines-release/raw/master/model/_trained_MPI/pose_deploy_resize.prototxt
# convert the model to a dict:
python -m tensorpack.utils.loadcaffe pose_deploy_resize.prototxt pose_iter_320000.caffemodel CPM-original.npy
# run it on a test image, and produce output.jpg
python load-cpm.py --load CPM-original.npy --input test.jpg
```
