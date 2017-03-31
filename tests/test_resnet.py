from case_script import TestPythonScript
import os
import numpy as np
import cv2
import shutil
import tarfile
import errno
from six.moves import urllib
import sys


CAFFE_ILSVRC12_URL = "http://dl.caffe.berkeleyvision.org/caffe_ilsvrc12.tar.gz"


# this is duplicated code, but this file should be self-contained
def mkdir_p(dirname):
    """ Make a dir recursively, but do nothing if the dir exists

    Args:
        dirname(str):
    """
    assert dirname is not None
    if dirname == '' or os.path.isdir(dirname):
        return
    try:
        os.makedirs(dirname)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise e


# this is duplicated code, but this file should be self-contained
def download(url, dir):
    """
    Download URL to a directory. Will figure out the filename automatically
    from URL.
    """
    mkdir_p(dir)
    fname = url.split('/')[-1]
    fpath = os.path.join(dir, fname)

    def _progress(count, block_size, total_size):
        sys.stdout.write('\r>> Downloading %s %.1f%%' %
                         (fname,
                             min(float(count * block_size) / total_size,
                                 1.0) * 100.0))
        sys.stdout.flush()
    try:
        fpath, _ = urllib.request.urlretrieve(url, fpath, reporthook=_progress)
        statinfo = os.stat(fpath)
        size = statinfo.st_size
    except Exception as e:
        raise e
    assert size > 0, "Download an empty file!"
    sys.stdout.write('\n')
    # TODO human-readable size
    print('Succesfully downloaded ' + fname + " " + str(size) + ' bytes.')
    return fpath


def fake_ilsvrc12():

    def mkdir(n):
        if not os.path.exists(n):
            os.mkdir(n)

    # create directories
    mkdir('ilsvrc_metadata')
    mkdir('ilsvrc_metadata/train')
    mkdir('ilsvrc_metadata/train/n02134418')
    mkdir('ilsvrc_metadata/train/n02134419')

    mkdir('ilsvrc_metadata/val')
    mkdir('ilsvrc_metadata/test')

    # image is some random noise
    def fake_image(shape=(255, 225, 3)):
        return np.random.randint(0, 255, shape)

    for subset in ['02134418', '02134419']:
        for i in range(2):
            fn = 'ilsvrc_metadata/train/n%s/n%s_%i.JPEG' % (subset, subset, i)
            cv2.imwrite(fn, fake_image())
        cv2.imwrite('ilsvrc_metadata/val/ILSVRC2012_val_%s.JPEG' % (subset), fake_image())
        cv2.imwrite('ilsvrc_metadata/test/ILSVRC2012_test_%s.JPEG' % (subset), fake_image())

    # download caffe_ilsvrc12.tar.gz if not cached yet
    tp_data_dir = os.path.join(os.environ['HOME'], 'tensorpack_data')
    if not os.path.isfile(os.path.join(tp_data_dir, 'caffe_ilsvrc12.tar.gz')):
        fpath = download(CAFFE_ILSVRC12_URL, tp_data_dir)
    else:
        print('caffe_ilsvrc12.tar.gz already exists')
        fpath = os.path.join(tp_data_dir, 'caffe_ilsvrc12.tar.gz')
    tarfile.open(fpath, 'r:gz').extractall('ilsvrc_metadata')

    # the ILSVRC dataset class uses these files to get the image paths
    with open('ilsvrc_metadata/train.txt', 'w') as f:
        f.write('n02134418/n02134418_0.JPEG 0\n')
        f.write('n02134418/n02134418_1.JPEG 0\n')
        f.write('n02134419/n02134419_0.JPEG 0\n')
        f.write('n02134419/n02134419_1.JPEG 0\n')

    with open('ilsvrc_metadata/val.txt', 'w') as f:
        f.write('ILSVRC2012_val_02134418.JPEG 0\n')
        f.write('ILSVRC2012_val_02134419.JPEG 0\n')

    with open('ilsvrc_metadata/test.txt', 'w') as f:
        f.write('ILSVRC2012_test_02134418.JPEG 0\n')
        f.write('ILSVRC2012_test_02134419.JPEG 0\n')


class ResnetTest(TestPythonScript):

    @property
    def script(self):
        return '../examples/ResNet/imagenet-resnet.py'

    def setUp(self):
        super(ResnetTest, self).setUp()
        fake_ilsvrc12()

    def test(self):
        self.assertSurvive(self.script, args=['--data ilsvrc_metadata', '--gpu 0'], timeout=10)

    def tearDown(self):
        super(ResnetTest, self).tearDown()
        if os.path.isdir('ilsvrc'):
            shutil.rmtree('ilsvrc')
