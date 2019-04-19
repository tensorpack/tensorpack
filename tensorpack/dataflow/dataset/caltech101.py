# -*- coding: utf-8 -*-
# File: mnist.py


import numpy
import os
import scipy.io
from six.moves import range

from ...utils import logger
from ...utils.fs import download, get_dataset_path
from ..base import RNGDataFlow

__all__ = ["Caltech101"]


def maybe_download(url, work_directory):
    """Download the data from Marlin's website, unless it's already here."""
    filename = url.split("/")[-1]
    filepath = os.path.join(work_directory, filename)
    if not os.path.exists(filepath):
        logger.info("Downloading to {}...".format(filepath))
        download(url, work_directory)
    return filepath


class Caltech101(RNGDataFlow):
    """
    Produces [image, label] in Caltech101 dataset,
    image is 28x28 in the range [0,1], label is an int.
    """

    _DIR_NAME = "caltech101_data"
    _SOURCE_URL = "https://people.cs.umass.edu/~marlin/data/"

    def __init__(self, train_or_test, shuffle=True, dir=None):
        """
        Args:
            train_or_test (str): either 'train' or 'test'
            shuffle (bool): shuffle the dataset
        """
        if dir is None:
            dir = get_dataset_path(self._DIR_NAME)
        assert train_or_test in ["train", "test"]
        self.train_or_test = train_or_test
        self.shuffle = shuffle

        def get_images_and_labels(data_file):
            f = maybe_download(self._SOURCE_URL + data_file, dir)
            data = scipy.io.loadmat(f)
            return data

        self.data = get_images_and_labels("caltech101_silhouettes_28_split1.mat")

        if self.train_or_test == "train":
            train_with_val_images = (self.data["train_data"], self.data["val_data"])
            self.images = numpy.concatenate(train_with_val_images, axis=0)
            self.images = self.images.reshape((6364, 28, 28))
            train_with_val_labels = (self.data["train_labels"], self.data["val_labels"])
            self.labels = numpy.concatenate(train_with_val_labels, axis=0).ravel() - 1
        else:
            self.images = self.data["test_data"].reshape((2307, 28, 28))
            self.labels = self.data["test_labels"].ravel() - 1

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


if __name__ == "__main__":
    ds = Caltech101("train")
    ds.reset_state()
    for (img, label) in ds:
        from IPython import embed

        embed()
        break
