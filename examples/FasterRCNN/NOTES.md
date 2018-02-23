
### File Structure
This is a minimal implementation that simply contains these files:
+ coco.py: load COCO data
+ data.py: prepare data for training
+ common.py: common data preparation utilities
+ basemodel.py: implement resnet
+ model.py: implement rpn/faster-rcnn/mask-rcnn
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

4. Because of (3), BatchNorm statistics are not supposed to be updated during fine-tuning.
	 This specific kind of BatchNorm will need [my kernel](https://github.com/tensorflow/tensorflow/pull/12580)
	 which is included since TF 1.4. If using an earlier version of TF, it will be either slow or wrong.

Speed:

1. The training will start very slow due to convolution warmup, until about 3k steps to reach a maximum speed.
	 Then the training speed will slowly decrease due to more accurate proposals.

2. Inference is not quite fast, because either you disable convolution autotune and end up with
	 a slow convolution algorithm, or you spend more time on autotune.
	 This is a general problem of TensorFlow when running against variable-sized input.

3. With a large roi batch size (e.g. >= 256), GPU utilitization should stay above 90%.

