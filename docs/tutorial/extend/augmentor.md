

### Write an Image Augmentor

The first thing to note: __you never have to write an augmentor__.
An augmentor is a part of the DataFlow, so you can always
[write a DataFlow](dataflow.html)
to do whatever operations to your data, rather than writing an augmentor.

Augmentor makes things easier when what you want fits its design.
But remember it is just an abstraction that may not always work for your use case.
For example, if your data transformation depend on multiple dataflow components,
or if you want to apply different transformations to different components,
the abstraction is often not enough for you, and you need to write code on the
DataFlow level instead.

An image augmentor maps an image to an image.
If you have such a mapping function `f` already, you can simply use
[imgaug.MapImage(f)](../../modules/dataflow.imgaug.html#tensorpack.dataflow.imgaug.MapImage)
as the augmentor, or use
[MapDataComponent(dataflow, f, index)](../../modules/dataflow.html#tensorpack.dataflow.MapDataComponent)
as the DataFlow.
In other words, for simple mapping you do not need to write an augmentor.

An augmentor may do something more than just applying a mapping.
To do complicated augmentation, the interface you will need to implement is:

```python
class MyAug(imgaug.ImageAugmentor):
  def _get_augment_params(self, img):
    # Generated random params with self.rng
    return params

  def _augment(self, img, params):
    return augmented_img

  # optional method
  def _augment_coords(self, coords, param):
    # coords is a Nx2 floating point array, each row is (x, y)
    return augmented_coords
```


#### The Design of imgaug Module

The [imgaug module](../../modules/dataflow.imgaug.html) is designed to allow the following usage:

* Factor out randomness and determinism.
  An augmentor may be randomized, but you can call
  [augment_return_params](../../modules/dataflow.imgaug.html#tensorpack.dataflow.imgaug.Augmentor.augment_return_params)
  to obtain the randomized parameters and then call
  [augment_with_params](../../modules/dataflow.imgaug.html#tensorpack.dataflow.imgaug.Augmentor.augment_with_params)
  on other data with the same randomized parameters.

* Because of the above reason, tensorpack's augmentor can augment multiple images together
  easily. This is commonly used for augmenting an image together with its masks.

* An image augmentor (e.g. flip) may also augment a coordinate, with
  [augment_coords](../../modules/dataflow.imgaug.html#tensorpack.dataflow.imgaug.ImageAugmentor.augment_coords).
  In this way, images can be augmented together with
  boxes, polygons, keypoints, etc.
  Coordinate augmentation enforces floating points coordinates
  to avoid quantization error.

* Reset random seed. Random seed can be reset by
  [reset_state](../../modules/dataflow.imgaug.html#tensorpack.dataflow.imgaug.Augmentor.reset_state).
  This is important for multi-process data loading, to make sure different
  processes get different seeds. 
  The reset method is called automatically if you use tensorpack's 
  [image augmentation dataflow](../../modules/dataflow.html#tensorpack.dataflow.AugmentImageComponent).
  Otherwise, **you are responsible** for calling it by yourself in subprocesses.
  See the
  [API documentation](../../modules/dataflow.imgaug.html#tensorpack.dataflow.imgaug.Augmentor.reset_state)
  of this method for more details.
