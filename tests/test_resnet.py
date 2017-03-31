from case_script import TestPythonScript
import os
import shutil


class ResnetTest(TestPythonScript):

    @property
    def script(self):
        return '../examples/ResNet/imagenet-resnet.py'

    def test(self):
        self.assertSurvive(self.script, args=['--data .',
                                              '--gpu 0', '--fake', '--data_format NHWC'], timeout=10)

    def tearDown(self):
        super(ResnetTest, self).tearDown()
        if os.path.isdir('ilsvrc'):
            shutil.rmtree('ilsvrc')
