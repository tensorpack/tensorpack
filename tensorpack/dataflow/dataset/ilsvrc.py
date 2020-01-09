# -*- coding: utf-8 -*-
# File: ilsvrc.py

import numpy as np
import os
import tarfile
import tqdm

from ...utils import logger
from ...utils.fs import download, get_dataset_path, mkdir_p
from ...utils.loadcaffe import get_caffe_pb
from ...utils.timer import timed_operation
from ..base import RNGDataFlow

__all__ = ['ILSVRCMeta', 'ILSVRC12', 'ILSVRC12Files']

CAFFE_ILSVRC12_URL = ("http://dl.caffe.berkeleyvision.org/caffe_ilsvrc12.tar.gz", 17858008)


class ILSVRCMeta(object):
    """
    Provide methods to access metadata for ILSVRC dataset.
    """

    def __init__(self, dir=None):
        if dir is None:
            dir = get_dataset_path('ilsvrc_metadata')
        self.dir = os.path.expanduser(dir)
        mkdir_p(self.dir)
        f = os.path.join(self.dir, 'synsets.txt')
        if not os.path.isfile(f):
            self._download_caffe_meta()
        self.caffepb = None

    def get_synset_words_1000(self):
        """
        Returns:
            dict: {cls_number: cls_name}
        """
        fname = os.path.join(self.dir, 'synset_words.txt')
        assert os.path.isfile(fname), fname
        lines = [x.strip() for x in open(fname).readlines()]
        return dict(enumerate(lines))

    def get_synset_1000(self):
        """
        Returns:
            dict: {cls_number: synset_id}
        """
        fname = os.path.join(self.dir, 'synsets.txt')
        assert os.path.isfile(fname)
        lines = [x.strip() for x in open(fname).readlines()]
        return dict(enumerate(lines))

    def _download_caffe_meta(self):
        fpath = download(CAFFE_ILSVRC12_URL[0], self.dir, expect_size=CAFFE_ILSVRC12_URL[1])
        tarfile.open(fpath, 'r:gz').extractall(self.dir)

    def get_image_list(self, name, dir_structure='original'):
        """
        Args:
            name (str): 'train' or 'val' or 'test'
            dir_structure (str): same as in :meth:`ILSVRC12.__init__()`.
        Returns:
            list: list of (image filename, label)
        """
        assert name in ['train', 'val', 'test']
        assert dir_structure in ['original', 'train']
        add_label_to_fname = (name != 'train' and dir_structure != 'original')
        if add_label_to_fname:
            synset = self.get_synset_1000()

        fname = os.path.join(self.dir, name + '.txt')
        assert os.path.isfile(fname), fname
        with open(fname) as f:
            ret = []
            for line in f.readlines():
                name, cls = line.strip().split()
                cls = int(cls)

                if add_label_to_fname:
                    name = os.path.join(synset[cls], name)

                ret.append((name.strip(), cls))
        assert len(ret), fname
        return ret

    def get_per_pixel_mean(self, size=None):
        """
        Args:
            size (tuple): image size in (h, w). Defaults to (256, 256).
        Returns:
            np.ndarray: per-pixel mean of shape (h, w, 3 (BGR)) in range [0, 255].
        """
        if self.caffepb is None:
            self.caffepb = get_caffe_pb()
        obj = self.caffepb.BlobProto()

        mean_file = os.path.join(self.dir, 'imagenet_mean.binaryproto')
        with open(mean_file, 'rb') as f:
            obj.ParseFromString(f.read())
        arr = np.array(obj.data).reshape((3, 256, 256)).astype('float32')
        arr = np.transpose(arr, [1, 2, 0])
        if size is not None:
            arr = cv2.resize(arr, size[::-1])
        return arr

    @staticmethod
    def guess_dir_structure(dir):
        """
        Return the directory structure of "dir".

        Args:
            dir(str): something like '/path/to/imagenet/val'

        Returns:
            either 'train' or 'original'
        """
        subdir = os.listdir(dir)[0]
        # find a subdir starting with 'n'
        if subdir.startswith('n') and \
                os.path.isdir(os.path.join(dir, subdir)):
            dir_structure = 'train'
        else:
            dir_structure = 'original'
        logger.info(
            "[ILSVRC12] Assuming directory {} has '{}' structure.".format(
                dir, dir_structure))
        return dir_structure


class ILSVRC12Files(RNGDataFlow):
    """
    Same as :class:`ILSVRC12`, but produces filenames of the images instead of nparrays.
    This could be useful when ``cv2.imread`` is a bottleneck and you want to
    decode it in smarter ways (e.g. in parallel).
    """
    def __init__(self, dir, name, meta_dir=None,
                 shuffle=None, dir_structure=None):
        """
        Same as in :class:`ILSVRC12`.
        """
        assert name in ['train', 'test', 'val'], name
        dir = os.path.expanduser(dir)
        assert os.path.isdir(dir), dir
        self.full_dir = os.path.join(dir, name)
        self.name = name
        assert os.path.isdir(self.full_dir), self.full_dir
        assert meta_dir is None or os.path.isdir(meta_dir), meta_dir
        if shuffle is None:
            shuffle = name == 'train'
        self.shuffle = shuffle

        if name == 'train':
            dir_structure = 'train'
        if dir_structure is None:
            dir_structure = ILSVRCMeta.guess_dir_structure(self.full_dir)

        meta = ILSVRCMeta(meta_dir)
        self.imglist = meta.get_image_list(name, dir_structure)

        for fname, _ in self.imglist[:10]:
            fname = os.path.join(self.full_dir, fname)
            assert os.path.isfile(fname), fname

    def __len__(self):
        return len(self.imglist)

    def __iter__(self):
        idxs = np.arange(len(self.imglist))
        if self.shuffle:
            self.rng.shuffle(idxs)
        for k in idxs:
            fname, label = self.imglist[k]
            fname = os.path.join(self.full_dir, fname)
            yield [fname, label]


class ILSVRC12(ILSVRC12Files):
    """
    Produces uint8 ILSVRC12 images of shape [h, w, 3(BGR)], and a label between [0, 999].
    The label map follows the synsets.txt file in http://dl.caffe.berkeleyvision.org/caffe_ilsvrc12.tar.gz.
    """
    def __init__(self, dir, name, meta_dir=None,
                 shuffle=None, dir_structure=None):
        """
        Args:
            dir (str): A directory containing a subdir named ``name``,
                containing the images in a structure described below.
            name (str): One of 'train' or 'val' or 'test'.
            shuffle (bool): shuffle the dataset.
                Defaults to True if name=='train'.
            dir_structure (str): One of 'original' or 'train'.
                The directory structure for the 'val' directory.
                'original' means the original decompressed directory, which only has list of image files (as below).
                If set to 'train', it expects the same two-level directory structure similar to 'dir/train/'.
                By default, it tries to automatically detect the structure.
                You probably do not need to care about this option because 'original' is what people usually have.

        Example:

        When `dir_structure=='original'`, `dir` should have the following structure:

        .. code-block:: none

            dir/
              train/
                n02134418/
                  n02134418_198.JPEG
                  ...
                ...
              val/
                ILSVRC2012_val_00000001.JPEG
                ...
              test/
                ILSVRC2012_test_00000001.JPEG
                ...

        With the downloaded ILSVRC12_img_*.tar, you can use the following
        command to build the above structure:

        .. code-block:: none

            mkdir val && tar xvf ILSVRC12_img_val.tar -C val
            mkdir test && tar xvf ILSVRC12_img_test.tar -C test
            mkdir train && tar xvf ILSVRC12_img_train.tar -C train && cd train
            find -type f -name '*.tar' | parallel -P 10 'echo {} && mkdir -p {/.} && tar xf {} -C {/.}'

        When `dir_structure=='train'`, `dir` should have the following structure:

        .. code-block:: none

            dir/
              train/
                n02134418/
                  n02134418_198.JPEG
                  ...
                ...
              val/
                n01440764/
                  ILSVRC2012_val_00000293.JPEG
                  ...
                ...
              test/
                ILSVRC2012_test_00000001.JPEG
                ...
        """
        super(ILSVRC12, self).__init__(
            dir, name, meta_dir, shuffle, dir_structure)

    """
    There are some CMYK / png images, but cv2 seems robust to them.
    https://github.com/tensorflow/models/blob/c0cd713f59cfe44fa049b3120c417cc4079c17e3/research/inception/inception/data/build_imagenet_data.py#L264-L300
    """
    def __iter__(self):
        for fname, label in super(ILSVRC12, self).__iter__():
            im = cv2.imread(fname, cv2.IMREAD_COLOR)
            assert im is not None, fname
            yield [im, label]

    @staticmethod
    def get_training_bbox(bbox_dir, imglist):
        import xml.etree.ElementTree as ET
        ret = []

        def parse_bbox(fname):
            root = ET.parse(fname).getroot()
            size = root.find('size').getchildren()
            size = map(int, [size[0].text, size[1].text])

            box = root.find('object').find('bndbox').getchildren()
            box = map(lambda x: float(x.text), box)
            return np.asarray(box, dtype='float32')

        with timed_operation('Loading Bounding Boxes ...'):
            cnt = 0
            for k in tqdm.trange(len(imglist)):
                fname = imglist[k][0]
                fname = fname[:-4] + 'xml'
                fname = os.path.join(bbox_dir, fname)
                try:
                    ret.append(parse_bbox(fname))
                    cnt += 1
                except Exception:
                    ret.append(None)
            logger.info("{}/{} images have bounding box.".format(cnt, len(imglist)))
        return ret


try:
    import cv2
except ImportError:
    from ...utils.develop import create_dummy_class
    ILSVRC12 = create_dummy_class('ILSVRC12', 'cv2')  # noqa

if __name__ == '__main__':
    meta = ILSVRCMeta()
    # print(meta.get_synset_words_1000())

    ds = ILSVRC12('/home/wyx/data/fake_ilsvrc/', 'train', shuffle=False)
    ds.reset_state()

    for _ in ds:
        from IPython import embed
        embed()
        break
