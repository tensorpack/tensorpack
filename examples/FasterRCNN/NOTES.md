
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
   + If you load a COCO-trained model on a different dataset, you'll see error messages
     complaining about unmatched number of categories for certain weights in the checkpoint.
     You can either remove those weights in checkpoint, or rename them in the model.
     See [tensorpack tutorial](https://tensorpack.readthedocs.io/tutorial/save-load.html) for more details.

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
   CuDNN warmup is by default on when no scale augmentation is used.

1. After warmup, the training speed will slowly decrease due to more accurate proposals.

1. The code should have around 70% GPU utilization on V100s, and 85%~90% scaling
   efficiency from 1 V100 to 8 V100s.

1. This implementation does not use specialized CUDA ops (e.g. AffineChannel, ROIAlign).
   Therefore it might be slower than other highly-optimized implementations.

Possible Future Enhancements:

1. Define a better interface to load different datasets.

1. Support batch>1 per GPU. Batching with inconsistent shapes is
   non-trivial to implement in TensorFlow.

1. Use dedicated ops to improve speed. (e.g. a TF implementation of ROIAlign op
   can be found in [light-head RCNN](https://github.com/zengarden/light_head_rcnn/tree/master/lib/lib_kernel))


### TensorFlow version notes

TensorFlow ≥ 1.6 supports most common features in this R-CNN implementation.
However, each version of TensorFlow has bugs that I either reported or fixed,
and this implementation touches many of those bugs.
Therefore, not every version of TF ≥ 1.6 supports every feature in this implementation.

1. TF < 1.6: Nothing works due to lack of support for empty tensors
   ([PR](https://github.com/tensorflow/tensorflow/pull/15264))
   and `FrozenBN` training
   ([PR](https://github.com/tensorflow/tensorflow/pull/12580)).
1. TF < 1.10: `SyncBN` with NCCL will fail ([PR](https://github.com/tensorflow/tensorflow/pull/20360)).
1. TF 1.11 & 1.12: multithread inference will fail ([issue](https://github.com/tensorflow/tensorflow/issues/22750)).
   Latest tensorpack will apply a workaround.
1. TF 1.13: MKL inference will fail ([issue](https://github.com/tensorflow/tensorflow/issues/24650)).
1. TF > 1.12: Horovod training will fail ([issue](https://github.com/tensorflow/tensorflow/issues/25946)).
   Latest tensorpack will apply a workaround.

This implementation contains workaround for some of these TF bugs.
However, note that the workaround needs to check your TF version by `tf.VERSION`,
and may not detect bugs properly if your TF version is not an official release
(e.g., if you use a nightly build).
