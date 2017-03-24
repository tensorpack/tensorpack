from case_script import TestPythonScript
import os
import numpy as np
import cv2
import shutil


def fake_ilsvrc12():

    def mkdir(n):
        if not os.path.exists(n):
            os.mkdir(n)

    mkdir('ilsvrc')
    mkdir('ilsvrc/train')
    mkdir('ilsvrc/train/n02134418')
    mkdir('ilsvrc/train/n02134419')

    mkdir('ilsvrc/val')
    mkdir('ilsvrc/val/n02134418')
    mkdir('ilsvrc/val/n02134419')

    def fake_image(shape=(255, 225, 3)):
        return np.random.randint(0, 255, shape)

    for subset in ['02134418', '02134419']:
        for i in range(2):
            fn = 'ilsvrc/train/n%s/n%s_%i.JPEG' % (subset, subset, i)
            cv2.imwrite(fn, fake_image())
            cv2.imwrite('ilsvrc/val/n%s/n%s_%i.JPEG' % (subset, subset, i), fake_image())


class ResnetTest(TestPythonScript):

    @property
    def script(self):
        return '../examples/ResNet/imagenet-resnet.py'

    def setUp(self):
        super(ResnetTest, self).setUp()
        fake_ilsvrc12()

    def test(self):
        self.assertSurvive(self.script, args=['--data ilsvrc', '--gpu 0'], timeout=10)

    def tearDown(self):
        super(ResnetTest, self).tearDown()
        if os.path.isdir('ilsvrc'):
            shutil.rmtree('ilsvrc')
