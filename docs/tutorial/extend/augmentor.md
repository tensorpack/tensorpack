
### Write an image augmentor

The first thing to note: an augmentor is a part of the DataFlow, so you can always
[write a DataFlow](http://tensorpack.readthedocs.io/en/latest/tutorial/extend/dataflow.html)
to do whatever operations to your data, rather than writing an augmentor.
Augmentors just sometimes make things easier.

An augmentor maps images to images.
If you have such a mapping function `f` already, you can simply use `imgaug.MapImage(f)` as the
augmentor, or use `MapDataComponent(df, f, index)` as the DataFlow.
In other words, for simple mapping you do not need to write an augmentor.

An augmentor may do something more than applying a mapping. The interface you will need to implement
is:

```python
class MyAug(imgaug.ImageAugmentor):
	def _get_augment_params(self, img):
		# generated random params with self.rng
	  return params

  def _augment(self, img, params):
	  return augmented_img
```

It does the following extra things for you:

1. `self.rng` is a `np.random.RandomState` object,
	guaranteed to have different seeds when you use multiprocess prefetch.
	In multiprocess settings, you have to use it to generate random numbers.

2. Random parameter generation and the actual augmentation is separated. This allows you to apply the
	same transformation to several images together (with `AugmentImageComponents`),
	which is essential to tasks such as segmentation.
