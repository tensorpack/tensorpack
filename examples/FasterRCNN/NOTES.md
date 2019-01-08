
### File Structure
This is a minimal implementation that simply contains these files:
+ dataset.py: load and evaluate COCO dataset
+ data.py: prepare data for training & inference
+ common.py: common data preparation utilities
+ basemodel.py: implement backbones
+ model_box.py: implement box-related symbolic functions
+ model_{fpn,rpn,frcnn,mrcnn,cascade}.py: implement FPN,RPN,Fast-/Mask-/Cascade-RCNN models.
+ train.py: main entry script
+ utils/: third-party helper functions
+ eval.py: evaluation utilities
+ viz.py: visualization utilities

### Implementation Notes

Data:

1. It's easy to train on your own data by changing `dataset.py`.

   + If your data is in COCO format, modify `COCODetection`
     to change the class names and the id mapping.
   + If your data is not in COCO format, ignore `COCODetection` completely and
     rewrite all the methods of
     `DetectionDataset` following its documents.
     You'll implement the logic to load your dataset and evaluate predictions.

2. You can easily add more augmentations such as rotation, but be careful how a box should be
	 augmented. The code now will always use the minimal axis-aligned bounding box of the 4 corners,
	 which is probably not the optimal way.
	 A TODO is to generate bounding box from segmentation, so more augmentations can be naturally supported.

Model:

1. Floating-point boxes are defined like this:

<p align="center"> <img src="https://user-images.githubusercontent.com/1381301/31527740-2f1b38ce-af84-11e7-8de1-628e90089826.png"> </p>

2. We use ROIAlign, and `tf.image.crop_and_resize` is __NOT__ ROIAlign.

3. We currently only support single image per GPU.

4. Because of (3), BatchNorm statistics are supposed to be freezed during fine-tuning.

5. An alternative to freezing BatchNorm is to sync BatchNorm statistics across
   GPUs (the `BACKBONE.NORM=SyncBN` option). This would require [my bugfix](https://github.com/tensorflow/tensorflow/pull/20360)
   which is available since TF 1.10. You can manually apply the patch to use it.
   For now the total batch size is at most 8, so this option does not improve the model by much.

6. Another alternative to BatchNorm is GroupNorm (`BACKBONE.NORM=GN`) which has better performance.

Speed:

1. If CuDNN warmup is on, the training will start very slowly, until about
   10k steps (or more if scale augmentation is used) to reach a maximum speed.
   As a result, the ETA is also inaccurate at the beginning.
   Warmup is by default on when no scale augmentation is used.

1. After warmup, the training speed will slowly decrease due to more accurate proposals.

1. The code should have around 70% GPU utilization on V100s, and 85%~90% scaling
   efficiency from 1 V100 to 8 V100s.

1. This implementation does not contain specialized CUDA ops (e.g. AffineChannel, ROIAlign),
   so it can be slightly (~10%) slower than Detectron (Caffe2) and
   maskrcnn-benchmark (PyTorch).

Possible Future Enhancements:

1. Define a better interface to load custom dataset.

1. Support batch>1 per GPU.

1. Use dedicated ops to improve speed. (e.g. a TF implementation of ROIAlign op
   can be found in [light-head RCNN](https://github.com/zengarden/light_head_rcnn/tree/master/lib/lib_kernel))
