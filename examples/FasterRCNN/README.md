# Faster R-CNN / Mask R-CNN on COCO
This example provides a minimal (2k lines) and faithful implementation of the
following object detection / instance segmentation papers:

+ [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/abs/1506.01497)
+ [Feature Pyramid Networks for Object Detection](https://arxiv.org/abs/1612.03144)
+ [Mask R-CNN](https://arxiv.org/abs/1703.06870)
+ [Cascade R-CNN: Delving into High Quality Object Detection](https://arxiv.org/abs/1712.00726)

with the support of:
+ Multi-GPU / distributed training, multi-GPU evaluation
+ Cross-GPU BatchNorm (aka Sync-BN, from [MegDet: A Large Mini-Batch Object Detector](https://arxiv.org/abs/1711.07240))
+ [Group Normalization](https://arxiv.org/abs/1803.08494)
+ Training from scratch (from [Rethinking ImageNet Pre-training](https://arxiv.org/abs/1811.08883))

This is likely the best-performing open source TensorFlow reimplementation of the above papers.

## Dependencies
+ Python 3.3+; OpenCV
+ TensorFlow ≥ 1.6
+ pycocotools: `pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'`
+ Pre-trained [ImageNet ResNet model](http://models.tensorpack.com/FasterRCNN/)
  from tensorpack model zoo
+ [COCO data](http://cocodataset.org/#download). It needs to have the following directory structure:
```
COCO/DIR/
  annotations/
    instances_train201?.json
    instances_val201?.json
  train201?/
    COCO_train201?_*.jpg
  val201?/
    COCO_val201?_*.jpg
```

You can use either the 2014 version or the 2017 version of the dataset.
To use the common "trainval35k + minival" split for the 2014 dataset, just
download the annotation files `instances_minival2014.json`,
`instances_valminusminival2014.json` from
[here](https://github.com/rbgirshick/py-faster-rcnn/blob/master/data/README.md)
to `annotations/` as well.

<sup>Note that train2017==trainval35k==train2014+val2014-minival2014, and val2017==minival2014</sup>


## Usage
### Train:

On a single machine:
```
./train.py --config \
    MODE_MASK=True MODE_FPN=True \
    DATA.BASEDIR=/path/to/COCO/DIR \
    BACKBONE.WEIGHTS=/path/to/ImageNet-R50-AlignPadding.npz
```

To run distributed training, set `TRAINER=horovod` and refer to [HorovodTrainer docs](http://tensorpack.readthedocs.io/modules/train.html#tensorpack.train.HorovodTrainer).

Options can be changed by either the command line or the `config.py` file (recommended).
Some reasonable configurations are listed in the table below.

### Inference:

To predict on an image (needs DISPLAY to show the outputs):
```
./train.py --predict input.jpg --load /path/to/Trained-Model-Checkpoint --config SAME-AS-TRAINING
```

To evaluate the performance of a model on COCO:
```
./train.py --evaluate output.json --load /path/to/Trained-Model-Checkpoint \
    --config SAME-AS-TRAINING
```

Several trained models can be downloaded in the table below. Evaluation and
prediction will need to be run with the corresponding configs used in training.

## Results

These models are trained on trainval35k and evaluated on minival2014 using mAP@IoU=0.50:0.95.
All models are fine-tuned from ImageNet pre-trained R50/R101 models in
[tensorpack model zoo](http://models.tensorpack.com/FasterRCNN/), unless otherwise noted.
All models are trained with 8 NVIDIA V100s, unless otherwise noted.

Performance in [Detectron](https://github.com/facebookresearch/Detectron/) can be roughly reproduced.

 | Backbone                    | mAP<br/>(box;mask)                                             | Detectron mAP <sup>[1](#ft1)</sup><br/> (box;mask) | Time (on 8 V100s) | Configurations <br/> (click to expand)                                                                                                                                                                                                                                                                                                                                                                        |
 | -                           | -                                                              | -                                                  | -                 | -                                                                                                                                                                                                                                                                                                                                                                                                             |
 | R50-C4                      | 33.5                                                           |                                                    | 17h               | <details><summary>super quick</summary>`MODE_MASK=False FRCNN.BATCH_PER_IM=64`<br/>`PREPROC.TRAIN_SHORT_EDGE_SIZE=600 PREPROC.MAX_SIZE=1024`<br/>`TRAIN.LR_SCHEDULE=[150000,230000,280000]` </details>                                                                                                                                                                                                        |
 | R50-C4                      | 36.6                                                           | 36.5                                               | 44h               | <details><summary>standard</summary>`MODE_MASK=False` </details>                                                                                                                                                                                                                                                                                                                                              |
 | R50-FPN                     | 37.4                                                           | 37.9                                               | 23h               | <details><summary>standard</summary>`MODE_MASK=False MODE_FPN=True` </details>                                                                                                                                                                                                                                                                                                                                |
 | R50-C4                      | 38.2;33.3 [:arrow_down:][R50C42x]                              | 37.8;32.8                                          | 49h               | <details><summary>standard</summary>this is the default </details>                                                                                                                                                                                                                                                                                                                                            |
 | R50-FPN                     | 38.4;35.1 [:arrow_down:][R50FPN2x]                             | 38.6;34.5                                          | 27h               | <details><summary>standard</summary>`MODE_FPN=True` </details>                                                                                                                                                                                                                                                                                                                                                |
 | R50-FPN                     | 42.0;36.3                                                      |                                                    | 36h               | <details><summary>+Cascade</summary>`MODE_FPN=True FPN.CASCADE=True` </details>                                                                                                                                                                                                                                                                                                                               |
 | R50-FPN                     | 39.5;35.2                                                      | 39.5;34.4<sup>[2](#ft2)</sup>                      | 31h               | <details><summary>+ConvGNHead</summary>`MODE_FPN=True`<br/>`FPN.FRCNN_HEAD_FUNC=fastrcnn_4conv1fc_gn_head` </details>                                                                                                                                                                                                                                                                                         |
 | R50-FPN                     | 40.0;36.2 [:arrow_down:][R50FPN2xGN]                           | 40.3;35.7                                          | 33h               | <details><summary>+GN</summary>`MODE_FPN=True`<br/>`FPN.NORM=GN BACKBONE.NORM=GN`<br/>`FPN.FRCNN_HEAD_FUNC=fastrcnn_4conv1fc_gn_head`<br/>`FPN.MRCNN_HEAD_FUNC=maskrcnn_up4conv_gn_head`                                                                                                                                                                                                                      |
 | R101-C4                     | 41.4;35.2 [:arrow_down:][R101C42x]                             |                                                    | 60h               | <details><summary>standard</summary>`BACKBONE.RESNET_NUM_BLOCKS=[3,4,23,3]` </details>                                                                                                                                                                                                                                                                                                                        |
 | R101-FPN                    | 40.4;36.6 [:arrow_down:][R101FPN2x]                            | 40.9;36.4                                          | 37h               | <details><summary>standard</summary>`MODE_FPN=True`<br/>`BACKBONE.RESNET_NUM_BLOCKS=[3,4,23,3]` </details>                                                                                                                                                                                                                                                                                                    |
 | R101-FPN                    | 46.5;40.1 [:arrow_down:][R101FPN3xCasAug] <sup>[3](#ft3)</sup> |                                                    | 73h               | <details><summary>3x+Cascade+TrainAug</summary>`MODE_FPN=True FPN.CASCADE=True`<br/>`BACKBONE.RESNET_NUM_BLOCKS=[3,4,23,3]`<br/>`TEST.RESULT_SCORE_THRESH=1e-4`<br/>`PREPROC.TRAIN_SHORT_EDGE_SIZE=[640,800]`<br/>`TRAIN.LR_SCHEDULE=[420000,500000,540000]` </details>                                                                                                                                       |
 | R101-FPN<br/>(From Scratch) | 47.5;41.2 [:arrow_down:][R101FPN9xGNCasAugScratch]             | 47.4;40.5<sup>[4](#ft4)</sup>                      | 45h (on 48 V100s) | <details><summary>9x+GN+Cascade+TrainAug</summary>`MODE_FPN=True FPN.CASCADE=True`<br/>`BACKBONE.RESNET_NUM_BLOCKS=[3,4,23,3]`<br/>`FPN.NORM=GN BACKBONE.NORM=GN`<br/>`FPN.FRCNN_HEAD_FUNC=fastrcnn_4conv1fc_gn_head`<br/>`FPN.MRCNN_HEAD_FUNC=maskrcnn_up4conv_gn_head`<br/>`PREPROC.TRAIN_SHORT_EDGE_SIZE=[640,800]`<br/>`TRAIN.LR_SCHEDULE=[1500000,1580000,1620000]`<br/>`BACKBONE.FREEZE_AT=0`</details> |
 
 [R50C42x]: http://models.tensorpack.com/FasterRCNN/COCO-R50C4-MaskRCNN-Standard.npz
 [R50FPN2x]: http://models.tensorpack.com/FasterRCNN/COCO-R50FPN-MaskRCNN-Standard.npz
 [R50FPN2xGN]: http://models.tensorpack.com/FasterRCNN/COCO-R50FPN-MaskRCNN-StandardGN.npz
 [R101C42x]: http://models.tensorpack.com/FasterRCNN/COCO-R101C4-MaskRCNN-Standard.npz
 [R101FPN2x]: http://models.tensorpack.com/FasterRCNN/COCO-R101FPN-MaskRCNN-Standard.npz
 [R101FPN3xCasAug]: http://models.tensorpack.com/FasterRCNN/COCO-R101FPN-MaskRCNN-BetterParams.npz
 [R101FPN9xGNCasAugScratch]: http://models.tensorpack.com/FasterRCNN/COCO-R101FPN-MaskRCNN-ScratchGN.npz

 <a id="ft1">1</a>: Numbers taken from [Detectron Model Zoo](https://github.com/facebookresearch/Detectron/blob/master/MODEL_ZOO.md).
 We compare models that have identical training & inference cost between the two implementations. However their numbers can be different due to many small implementation details.
For example, our FPN models are sometimes slightly worse in box AP, which is probably due to batch size.

 <a id="ft2">2</a>: Numbers taken from Table 5 in [Group Normalization](https://arxiv.org/abs/1803.08494)

 <a id="ft3">3</a>: Our mAP is __10+ point__ better than the official model in [matterport/Mask_RCNN](https://github.com/matterport/Mask_RCNN/releases/tag/v2.0) with the same R101-FPN backbone.

 <a id="ft4">4</a>: This entry does not use ImageNet pre-training. Detectron numbers are taken from Fig. 5 in [Rethinking ImageNet Pre-training](https://arxiv.org/abs/1811.08883).
 Note that our training strategy is slightly different: we enable cascade throughout the entire training.

## Notes

[NOTES.md](NOTES.md) has some notes about implementation details & speed.
