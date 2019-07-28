# -*- coding: utf-8 -*-
# File: meta.py


from .base import ImageAugmentor
from .transform import NoOpTransform, TransformList, TransformFactory

__all__ = ['RandomChooseAug', 'MapImage', 'Identity', 'RandomApplyAug',
           'RandomOrderAug']


class Identity(ImageAugmentor):
    """ A no-op augmentor """
    def get_transform(self, img):
        return NoOpTransform()


class RandomApplyAug(ImageAugmentor):
    """ Randomly apply the augmentor with a probability.
        Otherwise do nothing
    """

    def __init__(self, aug, prob):
        """
        Args:
            aug (ImageAugmentor): an augmentor.
            prob (float): the probability to apply the augmentor.
        """
        self._init(locals())
        super(RandomApplyAug, self).__init__()

    def get_transform(self, img):
        p = self.rng.rand()
        if p < self.prob:
            return self.aug.get_transform(img)
        else:
            return NoOpTransform()

    def reset_state(self):
        super(RandomApplyAug, self).reset_state()
        self.aug.reset_state()


class RandomChooseAug(ImageAugmentor):
    """ Randomly choose one from a list of augmentors """
    def __init__(self, aug_lists):
        """
        Args:
            aug_lists (list): list of augmentors, or list of (augmentor, probability) tuples
        """
        if isinstance(aug_lists[0], (tuple, list)):
            prob = [k[1] for k in aug_lists]
            aug_lists = [k[0] for k in aug_lists]
            self._init(locals())
        else:
            prob = [1.0 / len(aug_lists)] * len(aug_lists)
            self._init(locals())
        super(RandomChooseAug, self).__init__()

    def reset_state(self):
        super(RandomChooseAug, self).reset_state()
        for a in self.aug_lists:
            a.reset_state()

    def get_transform(self, img):
        aug_idx = self.rng.choice(len(self.aug_lists), p=self.prob)
        return self.aug_lists[aug_idx].get_transform(img)


class RandomOrderAug(ImageAugmentor):
    """
    Apply the augmentors with randomized order.
    """

    def __init__(self, aug_lists):
        """
        Args:
            aug_lists (list): list of augmentors.
                The augmentors are assumed to not change the shape of images.
        """
        self._init(locals())
        super(RandomOrderAug, self).__init__()

    def reset_state(self):
        super(RandomOrderAug, self).reset_state()
        for a in self.aug_lists:
            a.reset_state()

    def get_transform(self, img):
        # Note: this makes assumption that the augmentors do not make changes
        # to the image that will affect how the transforms will be instantiated
        # in the subsequent augmentors.
        idxs = self.rng.permutation(len(self.aug_lists))
        tfms = [self.aug_lists[k].get_transform(img)
                for k in range(len(self.aug_lists))]
        return TransformList([tfms[k] for k in idxs])


class MapImage(ImageAugmentor):
    """
    Map the image array by simple functions.
    """

    def __init__(self, func, coord_func=None):
        """
        Args:
            func: a function which takes an image array and return an augmented one
            coord_func: optional. A function which takes coordinates and return augmented ones.
                Coordinates should be Nx2 array of (x, y)s.
        """
        super(MapImage, self).__init__()
        self.func = func
        self.coord_func = coord_func

    def get_transform(self, img):
        if self.coord_func:
            return TransformFactory(name="MapImage", apply_image=self.func, apply_coords=self.coord_func)
        else:
            return TransformFactory(name="MapImage", apply_image=self.func)
