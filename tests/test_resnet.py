from case_script import TestPythonScript
import os
import shutil
import tarfile
from tensorpack.utils.fs import download, mkdir_p


CAFFE_ILSVRC12_URL = "http://dl.caffe.berkeleyvision.org/caffe_ilsvrc12.tar.gz"


def fake_ilsvrc12():

    # create directories
    mkdir_p('ilsvrc_metadata')

    # download caffe_ilsvrc12.tar.gz if not cached yet
    tp_data_dir = os.path.join(os.environ['HOME'], 'tensorpack_data')
    if not os.path.isfile(os.path.join(tp_data_dir, 'caffe_ilsvrc12.tar.gz')):
        fpath = download(CAFFE_ILSVRC12_URL, tp_data_dir)
    else:
        print('caffe_ilsvrc12.tar.gz already exists')
        fpath = os.path.join(tp_data_dir, 'caffe_ilsvrc12.tar.gz')
    tarfile.open(fpath, 'r:gz').extractall('ilsvrc_metadata')


class ResnetTest(TestPythonScript):

    @property
    def script(self):
        return '../examples/ResNet/imagenet-resnet.py'

    def setUp(self):
        super(ResnetTest, self).setUp()
        fake_ilsvrc12()

    def test(self):
        self.assertSurvive(self.script, args=['--data ilsvrc_metadata',
                                              '--gpu 0', '--fake True', '--data_format NHWC'], timeout=10)

    def tearDown(self):
        super(ResnetTest, self).tearDown()
        if os.path.isdir('ilsvrc'):
            shutil.rmtree('ilsvrc')
