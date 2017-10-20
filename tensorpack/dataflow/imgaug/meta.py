#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: meta.py
# Author: Yuxin Wu <ppwwyyxxc@gmail.com>


from .base import ImageAugmentor

__all__ = ['RandomChooseAug', 'MapImage', 'Identity', 'RandomApplyAug',
           'RandomOrderAug']


class Identity(ImageAugmentor):
    """ A no-op augmentor """
    def _augment(self, img, _):
        return img


class RandomApplyAug(ImageAugmentor):
    """ Randomly apply the augmentor with a probability.
        Otherwise do nothing
    """

    def __init__(self, aug, prob):
        """
        Args:
            aug (ImageAugmentor): an augmentor
            prob (float): the probability
        """
        self._init(locals())
        super(RandomApplyAug, self).__init__()

    def _get_augment_params(self, img):
        p = self.rng.rand()
        if p < self.prob:
            prm = self.aug._get_augment_params(img)
            return (True, prm)
        else:
            return (False, None)

    def _augment_return_params(self, img):
        p = self.rng.rand()
        if p < self.prob:
            img, prms = self.aug._augment_return_params(img)
            return img, (True, prms)
        else:
            return img, (False, None)

    def reset_state(self):
        super(RandomApplyAug, self).reset_state()
        self.aug.reset_state()

    def _augment(self, img, prm):
        if not prm[0]:
            return img
        else:
            return self.aug._augment(img, prm[1])

    def _augment_coords(self, coords, prm):
        if not prm[0]:
            return coords
        else:
            return self.aug._augment_coords(coords, prm[1])


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

    def _get_augment_params(self, img):
        aug_idx = self.rng.choice(len(self.aug_lists), p=self.prob)
        aug_prm = self.aug_lists[aug_idx]._get_augment_params(img)
        return aug_idx, aug_prm

    def _augment(self, img, prm):
        idx, prm = prm
        return self.aug_lists[idx]._augment(img, prm)

    def _augment_coords(self, coords, prm):
        idx, prm = prm
        return self.aug_lists[idx]._augment_coords(coords, prm)


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

    def _get_augment_params(self, img):
        # Note: If augmentors change the shape of image, get_augment_param might not work
        # All augmentors should only rely on the shape of image
        idxs = self.rng.permutation(len(self.aug_lists))
        prms = [self.aug_lists[k]._get_augment_params(img)
                for k in range(len(self.aug_lists))]
        return idxs, prms

    def _augment(self, img, prm):
        idxs, prms = prm
        for k in idxs:
            img = self.aug_lists[k]._augment(img, prms[k])
        return img

    def _augment_coords(self, coords, prm):
        idxs, prms = prm
        for k in idxs:
            img = self.aug_lists[k]._augment_coords(coords, prms[k])
        return img


class MapImage(ImageAugmentor):
    """
    Map the image array by a function.
    """

    def __init__(self, func, coord_func=None):
        """
        Args:
            func: a function which takes an image array and return an augmented one
        """
        super(MapImage, self).__init__()
        self.func = func
        self.coord_func = coord_func

    def _augment(self, img, _):
        return self.func(img)

    def _augment_coords(self, coords, _):
        if self.coord_func is None:
            raise NotImplementedError
        return self.coord_func(coords)
