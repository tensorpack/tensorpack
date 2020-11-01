#-*- coding: utf-8 -*-

import os
import numpy as np

from ...utils import logger
from ..base import RNGDataFlow


class Places365Standard(RNGDataFlow):
    """
    The Places365-Standard Dataset, in low resolution format only.
    Produces BGR images of shape (256, 256, 3) in range [0, 255].
    """
    def __init__(self, dir, name, shuffle=None):
        """
        Args:
            dir: path to the Places365-Standard dataset in its "easy directory
                structure".  See http://places2.csail.mit.edu/download.html
            name: one of "train" or "val"
            shuffle (bool): shuffle the dataset. Defaults to True if name=='train'.
        """
        assert name in ['train', 'val'], name
        dir = os.path.expanduser(dir)
        assert os.path.isdir(dir), dir
        self.name = name
        if shuffle is None:
            shuffle = name == 'train'
        self.shuffle = shuffle

        label_file = os.path.join(dir, name + ".txt")
        all_files = []
        labels = set()
        with open(label_file) as f:
            for line in f:
                filepath = os.path.join(dir, line.strip())
                line = line.strip().split("/")
                label = line[1]
                all_files.append((filepath, label))
                labels.add(label)
        self._labels = sorted(labels)
        # class ids are sorted alphabetically:
        # https://github.com/CSAILVision/places365/blob/master/categories_places365.txt
        labelmap = {label: id for id, label in enumerate(self._labels)}
        self._files = [(path, labelmap[x]) for path, x in all_files]
        logger.info("Found {} images in {}.".format(len(self._files), label_file))

    def get_label_names(self):
        """
        Returns:
            [str]: name of each class.
        """
        return self._labels

    def __len__(self):
        return len(self._files)

    def __iter__(self):
        idxs = np.arange(len(self._files))
        if self.shuffle:
            self.rng.shuffle(idxs)
        for k in idxs:
            fname, label = self._files[k]
            im = cv2.imread(fname, cv2.IMREAD_COLOR)
            assert im is not None, fname
            yield [im, label]


try:
    import cv2
except ImportError:
    from ...utils.develop import create_dummy_class
    Places365Standard = create_dummy_class('Places365Standard', 'cv2')  # noqa

if __name__ == '__main__':
    from tensorpack.dataflow import PrintData
    ds = Places365Standard("~/data/places365_standard/", 'train')
    ds = PrintData(ds, num=100)
    ds.reset_state()
    for k in ds:
        pass
