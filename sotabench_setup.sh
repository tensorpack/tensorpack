#!/bin/bash

pip install -e .

wget http://models.tensorpack.com/FasterRCNN/COCO-MaskRCNN-R101FPN9xGNCasAugScratch.npz

cd ./.data/vision/coco
unzip annotations_trainval2017.zip
unzip val2017.zip
cd -
