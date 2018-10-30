# Faster R-CNN / Mask R-CNN on COCO
This example provides a minimal (2k lines) and faithful implementation of the following papers:

+ [Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks](https://arxiv.org/abs/1506.01497)
+ [Feature Pyramid Networks for Object Detection](https://arxiv.org/abs/1612.03144)
+ [Mask R-CNN](https://arxiv.org/abs/1703.06870)
+ [Cascade R-CNN: Delving into High Quality Object Detection](https://arxiv.org/abs/1712.00726)

with the support of:
+ Multi-GPU / distributed training
+ [Cross-GPU BatchNorm](https://arxiv.org/abs/1711.07240)
+ [Group Normalization](https://arxiv.org/abs/1803.08494)

## Dependencies
+ Python 3; OpenCV.
+ TensorFlow >= 1.6 (1.4 or 1.5 can run but may crash due to a TF bug);
+ pycocotools: `pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'`
+ Pre-trained [ImageNet ResNet model](http://models.tensorpack.com/FasterRCNN/)
  from tensorpack model zoo.
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
    BACKBONE.WEIGHTS=/path/to/ImageNet-R50-AlignPadding.npz \
```

To run distributed training, set `TRAINER=horovod` and refer to [HorovodTrainer docs](http://tensorpack.readthedocs.io/modules/train.html#tensorpack.train.HorovodTrainer).

Options can be changed by either the command line or the `config.py` file.
Recommended configurations are listed in the table below.

The code is only valid for training with 1, 2, 4 or >=8 GPUs.
Not training with 8 GPUs may result in different performance from the table below.

### Inference:

To predict on an image (and show output in a window):
```
./train.py --predict input.jpg --load /path/to/model --config SAME-AS-TRAINING
```

To evaluate the performance of a model on COCO:
```
./train.py --evaluate output.json --load /path/to/COCO-R50C4-MaskRCNN-Standard.npz \
    --config SAME-AS-TRAINING
```

Several trained models can be downloaded in the table below. Evaluation and
prediction will need to be run with the corresponding training configs.

## Results

These models are trained on trainval35k and evaluated on minival2014 using mAP@IoU=0.50:0.95.
All models are fine-tuned from ImageNet pre-trained R50/R101 models in the [model zoo](http://models.tensorpack.com/FasterRCNN/).
Performance in [Detectron](https://github.com/facebookresearch/Detectron/) can be roughly reproduced.
Mask R-CNN results contain both box and mask mAP.

 | Backbone | mAP<br/>(box;mask)                                                                                                            | Detectron mAP <sup>[1](#ft1)</sup><br/> (box;mask) | Time on 8 V100s | Configurations <br/> (click to expand)                                                                                                                                                                                                                 |
 | -        | -                                                                                                                             | -                                                  | -               | -                                                                                                                                                                                                                                                      |
 | R50-C4   | 33.1                                                                                                                          |                                                    | 18h             | <details><summary>super quick</summary>`MODE_MASK=False FRCNN.BATCH_PER_IM=64`<br/>`PREPROC.SHORT_EDGE_SIZE=600 PREPROC.MAX_SIZE=1024`<br/>`TRAIN.LR_SCHEDULE=[150000,230000,280000]` </details>                                                       |
 | R50-C4   | 36.6                                                                                                                          | 36.5                                               | 44h             | <details><summary>standard</summary>`MODE_MASK=False` </details>                                                                                                                                                                                       |
 | R50-FPN  | 37.4                                                                                                                          | 37.9                                               | 29h             | <details><summary>standard</summary>`MODE_MASK=False MODE_FPN=True` </details>                                                                                                                                                                         |
 | R50-C4   | 38.2;33.3 [:arrow_down:](http://models.tensorpack.com/FasterRCNN/COCO-R50C4-MaskRCNN-Standard.npz)                            | 37.8;32.8                                          | 49h             | <details><summary>standard</summary>this is the default </details>                                                                                                                                                                                     |
 | R50-FPN  | 38.5;35.2 [:arrow_down:](http://models.tensorpack.com/FasterRCNN/COCO-R50FPN-MaskRCNN-Standard.npz)                           | 38.6;34.5                                          | 30h             | <details><summary>standard</summary>`MODE_FPN=True` </details>                                                                                                                                                                                         |
 | R50-FPN  | 42.0;36.3                                                                                                                     |                                                    | 41h             | <details><summary>+Cascade</summary>`MODE_FPN=True FPN.CASCADE=True` </details>                                                                                                                                                                        |
 | R50-FPN  | 39.5;35.2                                                                                                                     | 39.5;34.4<sup>[2](#ft2)</sup>                      | 33h             | <details><summary>+ConvGNHead</summary>`MODE_FPN=True`<br/>`FPN.FRCNN_HEAD_FUNC=fastrcnn_4conv1fc_gn_head` </details>                                                                                                                                  |
 | R50-FPN  | 40.0;36.2 [:arrow_down:](http://models.tensorpack.com/FasterRCNN/COCO-R50FPN-MaskRCNN-StandardGN.npz)                         | 40.3;35.7                                          | 40h             | <details><summary>+GN</summary>`MODE_FPN=True`<br/>`FPN.NORM=GN BACKBONE.NORM=GN`<br/>`FPN.FRCNN_HEAD_FUNC=fastrcnn_4conv1fc_gn_head`<br/>`FPN.MRCNN_HEAD_FUNC=maskrcnn_up4conv_gn_head`                                                               |
 | R101-C4  | 41.4;35.2 [:arrow_down:](http://models.tensorpack.com/FasterRCNN/COCO-R101C4-MaskRCNN-Standard.npz)                           |                                                    | 60h             | <details><summary>standard</summary>`BACKBONE.RESNET_NUM_BLOCK=[3,4,23,3]` </details>                                                                                                                                                                  |
 | R101-FPN | 40.4;36.6 [:arrow_down:](http://models.tensorpack.com/FasterRCNN/COCO-R101FPN-MaskRCNN-Standard.npz)                          | 40.9;36.4                                          | 38h             | <details><summary>standard</summary>`MODE_FPN=True`<br/>`BACKBONE.RESNET_NUM_BLOCK=[3,4,23,3]` </details>                                                                                                                                              |
 | R101-FPN | 46.5;40.1 [:arrow_down:](http://models.tensorpack.com/FasterRCNN/COCO-R101FPN-MaskRCNN-BetterParams.npz) <sup>[3](#ft3)</sup> |                                                    | 73h             | <details><summary>+++</summary>`MODE_FPN=True FPN.CASCADE=True`<br/>`BACKBONE.RESNET_NUM_BLOCK=[3,4,23,3]`<br/>`TEST.RESULT_SCORE_THRESH=1e-4`<br/>`PREPROC.TRAIN_SHORT_EDGE_SIZE=[640,800]`<br/>`TRAIN.LR_SCHEDULE=[420000,500000,540000]` </details> |

 <a id="ft1">1</a>: Here we comapre models that have identical training & inference cost between the two implementation. However their numbers are different due to many small implementation details.

 <a id="ft2">2</a>: Numbers taken from [Group Normalization](https://arxiv.org/abs/1803.08494)

 <a id="ft3">3</a>: Our mAP is __10+ point__ better than the official model in [matterport/Mask_RCNN](https://github.com/matterport/Mask_RCNN/releases/tag/v2.0) with the same R101-FPN backbone.

## Notes

[NOTES.md](NOTES.md) has some notes about implementation details & speed.
