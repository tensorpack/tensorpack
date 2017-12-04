import cv2
import os
import argparse
import numpy as np
import zipfile
import random
from tensorpack import RNGDataFlow, MapDataComponent, dftools


class ImageDataFromZIPFile(RNGDataFlow):
    """ Produce images read from a list of zip files. """
    def __init__(self, zip_file, shuffle=False, max_files=None):
        """
        Args:
            zip_file (list): list of zip file paths.
        """
        assert os.path.isfile(zip_file)
        self.shuffle = shuffle
        self.max = max_files
        self.archivefiles = []
        archive = zipfile.ZipFile(zip_file)
        imagesInArchive = archive.namelist()
        for img_name in imagesInArchive:
            if img_name.endswith('.jpg'):
                self.archivefiles.append((archive, img_name))
        if self.max is None:
            self.max = self.size()

    def size(self):
        return len(self.archivefiles)

    def get_data(self):
        if self.shuffle:
            self.rng.shuffle(self.archivefiles)
        self.archivefiles = random.sample(self.archivefiles, self.max)
        for archive in self.archivefiles:
            im_data = archive[0].read(archive[1])
            yield [im_data]


class ImageEncode(MapDataComponent):
    def __init__(self, ds, mode='.jpg', dtype=np.uint8, index=0):
        def func(img):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return np.asarray(bytearray(cv2.imencode(mode, img)[1].tostring()), dtype=dtype)
        super(ImageEncode, self).__init__(ds, func, index=index)


class ImageDecode(MapDataComponent):
    def __init__(self, ds, mode='.jpg', index=0):
        def func(im_data):
            img = cv2.imdecode(im_data, cv2.IMREAD_COLOR)
            return img
        super(ImageDecode, self).__init__(ds, func, index=index)


class RejectTooSmallImages(MapDataComponent):
    def __init__(self, ds, thresh=384, index=0):
        def func(img):
            h, w, _ = img.shape
            if (h < thresh) or (w < thresh):
                return None
            else:
                return img
        super(RejectTooSmallImages, self).__init__(ds, func, index=index)


class CenterSquareResize(MapDataComponent):
    def __init__(self, ds, index=0):
        """See section 5.3
        """
        def func(img):
            try:
                h, w, _ = img.shape
                if h > w:
                    off = (h - w) // 2
                    if off > 0:
                        img = img[off:-off, :, :]
                if w > h:
                    off = (w - h) // 2
                    if off > 0:
                        img = img[:, off:-off, :]

                img = cv2.resize(img, (256, 256))
                return img
            except Exception:
                return None
        super(CenterSquareResize, self).__init__(ds, func, index=index)


# Testcode for encode/decode.
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--create', action='store_true', help='create lmdb')
    parser.add_argument('--debug', action='store_true', help='debug images')
    parser.add_argument('--input', type=str, help='path to coco zip', required=True)
    parser.add_argument('--lmdb', type=str, help='path to output lmdb', required=True)
    args = parser.parse_args()

    ds = ImageDataFromZIPFile(args.input)
    ds = ImageDecode(ds, index=0)
    ds = RejectTooSmallImages(ds, index=0)
    ds = CenterSquareResize(ds, index=0)
    if args.create:
        ds = ImageEncode(ds, index=0)
        dftools.dump_dataflow_to_lmdb(ds, args.lmdb)
    if args.debug:
        ds.reset_state()
        for i in ds.get_data():
            cv2.imshow('example', i[0])
            cv2.waitKey(0)
