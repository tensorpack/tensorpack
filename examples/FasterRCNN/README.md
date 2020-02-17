# Faster R-CNN / Mask R-CNN on COCO
This example provides a minimal (2k lines) and faithful implementation of the
following object detection / instance segmentation papers:

+ [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/abs/1506.01497)
+ [Feature Pyramid Networks for Object Detection](https://arxiv.org/abs/1612.03144)
+ [Mask R-CNN](https://arxiv.org/abs/1703.06870)
+ [Cascade R-CNN: Delving into High Quality Object Detection](https://arxiv.org/abs/1712.00726)

with the support of:
+ Multi-GPU / multi-node distributed training, multi-GPU evaluation
+ Cross-GPU BatchNorm (aka Sync-BN, from [MegDet: A Large Mini-Batch Object Detector](https://arxiv.org/abs/1711.07240))
+ [Group Normalization](https://arxiv.org/abs/1803.08494)
+ Training from scratch (from [Rethinking ImageNet Pre-training](https://arxiv.org/abs/1811.08883))

This is likely the best-performing open source TensorFlow reimplementation of the above papers.

## Dependencies
+ OpenCV, TensorFlow ≥ 1.6
+ pycocotools/scipy: `for i in cython 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI' scipy; do pip install $i; done`
+ Pre-trained [ImageNet ResNet model](http://models.tensorpack.com/#FasterRCNN)
  from tensorpack model zoo
+ [COCO data](http://cocodataset.org/#download). It needs to have the following directory structure:
```
COCO/DIR/
  annotations/
    instances_train201?.json
    instances_val201?.json
  train201?/
    # image files that are mentioned in the corresponding json
  val201?/
    # image files that are mentioned in corresponding json
```

You can use either the 2014 version or the 2017 version of the dataset.
To use the common "trainval35k + minival" split for the 2014 dataset, just
download the annotation files `instances_minival2014.json`,
`instances_valminusminival2014.json` from
[here](https://github.com/rbgirshick/py-faster-rcnn/blob/master/data/README.md)
to `annotations/` as well.


## Usage

It is recommended to get familiar the relevant papers listed above before using this code.
Otherwise you may end up doing something unreasonable.

### Train:

To train on a single machine (with 1 or more GPUs):
```
./train.py --config \
    BACKBONE.WEIGHTS=/path/to/ImageNet-R50-AlignPadding.npz \
    DATA.BASEDIR=/path/to/COCO/DIR \
    [OTHER-ARCHITECTURE-SETTINGS]
```

Alternatively, use `TRAINER=horovod` which supports distributed training as well, but less straightforward to run.
Refer to [HorovodTrainer docs](http://tensorpack.readthedocs.io/modules/train.html#tensorpack.train.HorovodTrainer) for details.

All options can be changed by either the command line or the `config.py` file (recommended).
Some reasonable configurations are listed in the table below.
See [config.py](config.py) for details about how to correctly set `BACKBONE.WEIGHTS` and other configs.

### Inference:

To predict on given images (needs DISPLAY to show the outputs):
```
./predict.py --predict input1.jpg input2.jpg --load /path/to/Trained-Model-Checkpoint --config SAME-AS-TRAINING
```

To evaluate the performance of a model on COCO:
```
./predict.py --evaluate output.json --load /path/to/Trained-Model-Checkpoint \
    --config SAME-AS-TRAINING
```

Several trained models can be downloaded in the table below. Evaluation and
prediction have to be run with the corresponding configs used in training.

## Results

These models are trained on train2017 and evaluated on val2017 using mAP@IoU=0.50:0.95.
Unless otherwise noted, all models are fine-tuned from ImageNet pre-trained R50/R101 models in
[tensorpack model zoo](http://models.tensorpack.com/#FasterRCNN),
using 8 NVIDIA V100s.

Performance in [Detectron](https://github.com/facebookresearch/Detectron/) can be reproduced.

 | Backbone                       | mAP<br/>(box;mask)                                                      | Detectron mAP <sup>[1](#ft1)</sup><br/> (box;mask) | Time <br/>(on 8 V100s) | Configurations <br/> (click to expand)                                                                                                                                                                                                                                                                                                                                   |
 | -                              | -                                                                       | -                                                  | -                      | -                                                                                                                                                                                                                                                                                                                                                                        |
 | R50-FPN                        | 34.8                                                                    |                                                    | 6.5h                   | <details><summary>super quick</summary>`MODE_MASK=False FRCNN.BATCH_PER_IM=64`<br/>`PREPROC.TRAIN_SHORT_EDGE_SIZE=[500,800] PREPROC.MAX_SIZE=1024` </details>                                                                                                                                                                                                            |
 | R50-C4                         | 35.6                                                                    | 34.8                                               | 22.5h                  | <details><summary>standard</summary>`MODE_MASK=False MODE_FPN=False` </details>                                                                                                                                                                                                                                                                                          |
 | R50-FPN                        | 37.5                                                                    | 36.7                                               | 10.5h                  | <details><summary>standard</summary>`MODE_MASK=False` </details>                                                                                                                                                                                                                                                                                                         |
 | R50-C4                         | 36.2;31.8 [:arrow_down:][R50C41x]                                       | 35.8;31.4                                          | 23h                    | <details><summary>standard</summary>`MODE_FPN=False` </details>                                                                                                                                                                                                                                                                                                          |
 | R50-FPN                        | 38.2;34.8                                                               | 37.7;33.9                                          | 12.5h                  | <details><summary>standard</summary>this is the default </details>                                                                                                                                                                                                                                                                                                       |
 | R50-FPN                        | 38.9;35.4 [:arrow_down:][R50FPN2x]                                      | 38.6;34.5                                          | 24h                    | <details><summary>2x</summary>`TRAIN.LR_SCHEDULE=2x` </details>                                                                                                                                                                                                                                                                                                          |
 | R50-FPN-GN                     | 40.4;36.3 [:arrow_down:][R50FPN2xGN]                                    | 40.3;35.7                                          | 29h                    | <details><summary>2x+GN</summary>`FPN.NORM=GN BACKBONE.NORM=GN`<br/>`FPN.FRCNN_HEAD_FUNC=fastrcnn_4conv1fc_gn_head`<br/>`FPN.MRCNN_HEAD_FUNC=maskrcnn_up4conv_gn_head` <br/>`TRAIN.LR_SCHEDULE=2x`                                                                                                                                                                       |
 | R50-FPN                        | 41.7;36.2 [:arrow_down:][R50FPN1xCas]                                   |                                                    | 16h                    | <details><summary>+Cascade</summary>`FPN.CASCADE=True` </details>                                                                                                                                                                                                                                                                                                        |
 | R50-FPN-GN                     | 46.1;40.1 [:arrow_down:][R50FPN4xGNCasAug]                              |                                                    | 36h (on 16 V100s)      | <details><summary>4x+GN+Cascade+TrainAug</summary>`FPN.CASCADE=True`<br/>`FPN.NORM=GN BACKBONE.NORM=GN`<br/>`FPN.FRCNN_HEAD_FUNC=fastrcnn_4conv1fc_gn_head`<br/>`FPN.MRCNN_HEAD_FUNC=maskrcnn_up4conv_gn_head`<br/>`PREPROC.TRAIN_SHORT_EDGE_SIZE=[640,800]`<br/>`TRAIN.LR_SCHEDULE=4x` </details>                                                                       |
 | R101-C4                        | 40.1;34.6 [:arrow_down:][R101C41x]                                      |                                                    | 27h                    | <details><summary>standard</summary>`MODE_FPN=False`<br/>`BACKBONE.RESNET_NUM_BLOCKS=[3,4,23,3]` </details>                                                                                                                                                                                                                                                              |
 | R101-FPN                       | 40.7;36.8 [:arrow_down:][R101FPN1x] <sup>[2](#ft2)</sup>                | 40.0;35.9                                          | 17h                    | <details><summary>standard</summary>`BACKBONE.RESNET_NUM_BLOCKS=[3,4,23,3]` </details>                                                                                                                                                                                                                                                                                   |
 | R101-FPN                       | 46.6;40.3 [:arrow_down:][R101FPN3xCasAug]                               |                                                    | 64h                    | <details><summary>3x+Cascade+TrainAug</summary>` FPN.CASCADE=True`<br/>`BACKBONE.RESNET_NUM_BLOCKS=[3,4,23,3]`<br/>`TEST.RESULT_SCORE_THRESH=1e-4`<br/>`PREPROC.TRAIN_SHORT_EDGE_SIZE=[640,800]`<br/>`TRAIN.LR_SCHEDULE=3x` </details>                                                                                                                                   |
 | R101-FPN-GN<br/>(From Scratch) | 47.7;41.7 [:arrow_down:][R101FPN9xGNCasAugScratch] <sup>[3](#ft3)</sup> | 47.4;40.5                                          | 28h (on 64 V100s)      | <details><summary>9x+GN+Cascade+TrainAug</summary>`FPN.CASCADE=True`<br/>`BACKBONE.RESNET_NUM_BLOCKS=[3,4,23,3]`<br/>`FPN.NORM=GN BACKBONE.NORM=GN`<br/>`FPN.FRCNN_HEAD_FUNC=fastrcnn_4conv1fc_gn_head`<br/>`FPN.MRCNN_HEAD_FUNC=maskrcnn_up4conv_gn_head`<br/>`PREPROC.TRAIN_SHORT_EDGE_SIZE=[640,800]`<br/>`TRAIN.LR_SCHEDULE=9x`<br/>`BACKBONE.FREEZE_AT=0`</details> |

 [R50C41x]: http://models.tensorpack.com/FasterRCNN/COCO-MaskRCNN-R50C41x.npz
 [R50FPN2x]: http://models.tensorpack.com/FasterRCNN/COCO-MaskRCNN-R50FPN2x.npz
 [R50FPN2xGN]: http://models.tensorpack.com/FasterRCNN/COCO-MaskRCNN-R50FPN2xGN.npz
 [R50FPN1xCas]: http://models.tensorpack.com/FasterRCNN/COCO-MaskRCNN-R50FPN1xCas.npz
 [R50FPN4xGNCasAug]: http://models.tensorpack.com/FasterRCNN/COCO-MaskRCNN-R50FPN4xGNCasAug.npz
 [R101C41x]: http://models.tensorpack.com/FasterRCNN/COCO-MaskRCNN-R101C41x.npz
 [R101FPN1x]: http://models.tensorpack.com/FasterRCNN/COCO-MaskRCNN-R101FPN1x.npz
 [R101FPN3xCasAug]: http://models.tensorpack.com/FasterRCNN/COCO-MaskRCNN-R101FPN3xCasAug.npz
 [R101FPN9xGNCasAugScratch]: http://models.tensorpack.com/FasterRCNN/COCO-MaskRCNN-R101FPN9xGNCasAugScratch.npz

 <a id="ft1">1</a>: Numbers taken from [Detectron Model Zoo](https://github.com/facebookresearch/Detectron/blob/master/MODEL_ZOO.md).
 We compare models that have identical training & inference cost between the two implementations.
 Their numbers can be different due to small implementation details.

 <a id="ft2">2</a>: Our mAP is __7 point__ better than the official model in
 [matterport/Mask_RCNN](https://github.com/matterport/Mask_RCNN/releases/tag/v2.0) which has the same architecture.
 Our implementation is also [5x faster](https://github.com/tensorpack/benchmarks/tree/master/MaskRCNN).

 <a id="ft3">3</a>: This entry does not use ImageNet pre-training. Detectron numbers are taken from Fig. 5 in [Rethinking ImageNet Pre-training](https://arxiv.org/abs/1811.08883).
 Note that our training strategy is slightly different: we enable cascade throughout the entire training.
 As far as I know, this model is the __best open source TF model__ on COCO dataset.

## Use Custom Datasets / Implementation Details / Speed:

See [BALLOON.md](BALLOON.md) and [NOTES.md](NOTES.md) for more details.
