# -*- coding: utf-8 -*-
# File: caltech101.py


import os

from ...utils import logger
from ...utils.fs import download, get_dataset_path
from ..base import RNGDataFlow

__all__ = ["Caltech101Silhouettes"]


def maybe_download(url, work_directory):
    """Download the data from Marlin's website, unless it's already here."""
    filename = url.split("/")[-1]
    filepath = os.path.join(work_directory, filename)
    if not os.path.exists(filepath):
        logger.info("Downloading to {}...".format(filepath))
        download(url, work_directory)
    return filepath


class Caltech101Silhouettes(RNGDataFlow):
    """
    Produces [image, label] in Caltech101 Silhouettes dataset,
    image is 28x28 in the range [0,1], label is an int in the range [0,100].
    """

    _DIR_NAME = "caltech101_data"
    _SOURCE_URL = "https://people.cs.umass.edu/~marlin/data/"

    def __init__(self, name, shuffle=True, dir=None):
        """
        Args:
            name (str): 'train', 'test', 'val'
            shuffle (bool): shuffle the dataset
        """
        if dir is None:
            dir = get_dataset_path(self._DIR_NAME)
        assert name in ['train', 'test', 'val']
        self.name = name
        self.shuffle = shuffle

        def get_images_and_labels(data_file):
            f = maybe_download(self._SOURCE_URL + data_file, dir)
            data = scipy.io.loadmat(f)
            return data

        self.data = get_images_and_labels("caltech101_silhouettes_28_split1.mat")

        if self.name == "train":
            self.images = self.data["train_data"].reshape((4100, 28, 28))
            self.labels = self.data["train_labels"].ravel() - 1
        elif self.name == "test":
            self.images = self.data["test_data"].reshape((2307, 28, 28))
            self.labels = self.data["test_labels"].ravel() - 1
        else:
            self.images = self.data["val_data"].reshape((2264, 28, 28))
            self.labels = self.data["val_labels"].ravel() - 1

    def __len__(self):
        return self.images.shape[0]

    def __iter__(self):
        idxs = list(range(self.__len__()))
        if self.shuffle:
            self.rng.shuffle(idxs)
        for k in idxs:
            img = self.images[k]
            label = self.labels[k]
            yield [img, label]


try:
    import scipy.io
except ImportError:
    from ...utils.develop import create_dummy_class
    Caltech101Silhouettes = create_dummy_class('Caltech101Silhouettes', 'scipy.io') # noqa


if __name__ == "__main__":
    ds = Caltech101Silhouettes("train")
    ds.reset_state()
    for _ in ds:
        from IPython import embed

        embed()
        break
