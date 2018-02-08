from case_script import TestPythonScript

from tensorpack.tfutils.common import get_tf_version_number


class InfoGANTest(TestPythonScript):

    @property
    def script(self):
        return '../examples/GAN/InfoGAN-mnist.py'

    def test(self):
        if get_tf_version_number() < 1.4:
            return True     # requires leaky_relu
        self.assertSurvive(self.script, args=None)
