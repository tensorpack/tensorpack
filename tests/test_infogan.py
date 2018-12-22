from case_script import TestPythonScript

from tensorpack.tfutils.common import get_tf_version_tuple


class InfoGANTest(TestPythonScript):

    @property
    def script(self):
        return '../examples/GAN/InfoGAN-mnist.py'

    def test(self):
        return True  # https://github.com/tensorflow/tensorflow/issues/24517
        if get_tf_version_tuple() < (1, 4):
            return True     # requires leaky_relu
        self.assertSurvive(self.script, args=None)
