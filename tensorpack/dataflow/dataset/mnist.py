# -*- coding: utf-8 -*-
# File: mnist.py


import gzip
import numpy
import os

from ...utils import logger
from ...utils.fs import download, get_dataset_path
from ..base import RNGDataFlow

__all__ = ['Mnist', 'FashionMnist']


def maybe_download(url, work_directory):
    """Download the data from Yann's website, unless it's already here."""
    filename = url.split('/')[-1]
    filepath = os.path.join(work_directory, filename)
    if not os.path.exists(filepath):
        logger.info("Downloading to {}...".format(filepath))
        download(url, work_directory)
    return filepath


def _read32(bytestream):
    dt = numpy.dtype(numpy.uint32).newbyteorder('>')
    return numpy.frombuffer(bytestream.read(4), dtype=dt)[0]


def extract_images(filename):
    """Extract the images into a 4D uint8 numpy array [index, y, x, depth]."""
    with gzip.open(filename) as bytestream:
        magic = _read32(bytestream)
        if magic != 2051:
            raise ValueError(
                'Invalid magic number %d in MNIST image file: %s' %
                (magic, filename))
        num_images = _read32(bytestream)
        rows = _read32(bytestream)
        cols = _read32(bytestream)
        buf = bytestream.read(rows * cols * num_images)
        data = numpy.frombuffer(buf, dtype=numpy.uint8)
        data = data.reshape(num_images, rows, cols, 1)
        data = data.astype('float32') / 255.0
        return data


def extract_labels(filename):
    """Extract the labels into a 1D uint8 numpy array [index]."""
    with gzip.open(filename) as bytestream:
        magic = _read32(bytestream)
        if magic != 2049:
            raise ValueError(
                'Invalid magic number %d in MNIST label file: %s' %
                (magic, filename))
        num_items = _read32(bytestream)
        buf = bytestream.read(num_items)
        labels = numpy.frombuffer(buf, dtype=numpy.uint8)
        return labels


class Mnist(RNGDataFlow):
    """
    Produces [image, label] in MNIST dataset,
    image is 28x28 in the range [0,1], label is an int.
    """

    _DIR_NAME = 'mnist_data'
    _SOURCE_URL = 'http://yann.lecun.com/exdb/mnist/'

    def __init__(self, train_or_test, shuffle=True, dir=None):
        """
        Args:
            train_or_test (str): either 'train' or 'test'
            shuffle (bool): shuffle the dataset
        """
        if dir is None:
            dir = get_dataset_path(self._DIR_NAME)
        assert train_or_test in ['train', 'test']
        self.train_or_test = train_or_test
        self.shuffle = shuffle

        def get_images_and_labels(image_file, label_file):
            f = maybe_download(self._SOURCE_URL + image_file, dir)
            images = extract_images(f)
            f = maybe_download(self._SOURCE_URL + label_file, dir)
            labels = extract_labels(f)
            assert images.shape[0] == labels.shape[0]
            return images, labels

        if self.train_or_test == 'train':
            self.images, self.labels = get_images_and_labels(
                'train-images-idx3-ubyte.gz',
                'train-labels-idx1-ubyte.gz')
        else:
            self.images, self.labels = get_images_and_labels(
                't10k-images-idx3-ubyte.gz',
                't10k-labels-idx1-ubyte.gz')

    def __len__(self):
        return self.images.shape[0]

    def __iter__(self):
        idxs = list(range(self.__len__()))
        if self.shuffle:
            self.rng.shuffle(idxs)
        for k in idxs:
            img = self.images[k].reshape((28, 28))
            label = self.labels[k]
            yield [img, label]


class FashionMnist(Mnist):
    """
    Same API as :class:`Mnist`, but more fashion.
    """

    _DIR_NAME = 'fashion_mnist_data'
    _SOURCE_URL = 'http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/'

    def get_label_names(self):
        """
        Returns:
            [str]: the name of each class
        """
        # copied from https://github.com/zalandoresearch/fashion-mnist
        return ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


if __name__ == '__main__':
    ds = Mnist('train')
    ds.reset_state()
    for _ in ds:
        from IPython import embed
        embed()
        break
