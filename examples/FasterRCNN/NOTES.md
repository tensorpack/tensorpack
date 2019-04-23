
### File Structure
This is a minimal implementation that simply contains these files:
+ coco.py: load COCO data
+ data.py: prepare data for training
+ common.py: common data preparation utilities
+ basemodel.py: implement backbones
+ model_box.py: implement box-related symbolic functions
+ model_{fpn,rpn,mrcnn,frcnn}.py: implement FPN,RPN,Mask-/Fast-RCNN models.
+ train.py: main training script
+ utils/: third-party helper functions
+ eval.py: evaluation utilities
+ viz.py: visualization utilities

### Implementation Notes

Data:

1. It's easy to train on your own data. Just replace `COCODetection.load_many` in `data.py` by your own loader.
	Also remember to change `config.NUM_CLASS` and `config.CLASS_NAMES`.
	The current evaluation code is also COCO-specific, and you need to change it to use your data and metrics.

2. You can easily add more augmentations such as rotation, but be careful how a box should be
	 augmented. The code now will always use the minimal axis-aligned bounding box of the 4 corners,
	 which is probably not the optimal way.
	 A TODO is to generate bounding box from segmentation, so more augmentations can be naturally supported.

Model:

1. Floating-point boxes are defined like this:

<p align="center"> <img src="https://user-images.githubusercontent.com/1381301/31527740-2f1b38ce-af84-11e7-8de1-628e90089826.png"> </p>

2. We use ROIAlign, and because of (1), `tf.image.crop_and_resize` is __NOT__ ROIAlign.

3. We only support single image per GPU.

4. Because of (3), BatchNorm statistics are supposed to be freezed during fine-tuning.
   This specific kind of BatchNorm will need [my kernel](https://github.com/tensorflow/tensorflow/pull/12580)
   which is included since TF 1.4.
   
5. An alternative to freezing BatchNorm is to sync BatchNorm statistics across
   GPUs (the `BACKBONE.NORM=SyncBN` option). This would require [my bugfix](https://github.com/tensorflow/tensorflow/pull/20360)
   which will probably be in TF 1.10. You can manually apply the patch to use it.
   For now the total batch size is at most 8, so this option does not improve the model by much.

Speed:

1. The training will start very slow due to convolution warmup, until about 10k steps to reach a maximum speed.
   Then the training speed will slowly decrease due to more accurate proposals.

2. This implementation is about 14% slower than detectron,
   probably due to the lack of specialized ops (e.g. AffineChannel, ROIAlign) in TensorFlow.
   It's certainly faster than other TF implementation.

Possible Future Enhancements:

1. Data-parallel evaluation during training.

2. Define an interface to load custom dataset.

3. Support batch>1 per GPU.

4. Use dedicated ops to improve speed. (e.g. a TF implementation of ROIAlign op
   can be found in [light-head RCNN](https://github.com/zengarden/light_head_rcnn/tree/master/lib/lib_kernel))
